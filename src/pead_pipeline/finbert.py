from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


MODEL_ID = "ProsusAI/finbert"
MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
MAX_LENGTH = 512
STRIDE = 64
AGGREGATION = "new_token_weighted_mean_probability"


@dataclass(frozen=True)
class ScoreConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    max_length: int = MAX_LENGTH
    stride: int = STRIDE
    aggregation: str = AGGREGATION
    precision: str = "cuda_float16"
    event_batch_size: int = 8
    window_batch_size: int = 16

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SectionScore:
    positive: float
    negative: float
    neutral: float
    tone: float
    windows: int
    tokens: int


def compute_window_weights(
    content_lengths: Sequence[int],
    sample_mapping: Sequence[int],
    stride: int,
) -> list[int]:
    if len(content_lengths) != len(sample_mapping):
        raise ValueError("content lengths and sample mapping must have equal size")
    seen: set[int] = set()
    weights: list[int] = []
    for content_length, sample_index in zip(content_lengths, sample_mapping):
        if content_length <= 0:
            raise ValueError("every window must contain at least one content token")
        if sample_index in seen:
            weight = content_length - stride
            if weight <= 0:
                raise ValueError("overflow window contains no new content tokens")
        else:
            weight = content_length
            seen.add(sample_index)
        weights.append(weight)
    return weights


def aggregate_window_probabilities(
    probabilities: Sequence[Sequence[float]],
    sample_mapping: Sequence[int],
    content_lengths: Sequence[int],
    sample_count: int,
    stride: int,
    label_indices: dict[str, int],
) -> list[SectionScore]:
    weights = compute_window_weights(content_lengths, sample_mapping, stride)
    totals = [0 for _ in range(sample_count)]
    windows = [0 for _ in range(sample_count)]
    sums = [dict(positive=0.0, negative=0.0, neutral=0.0) for _ in range(sample_count)]
    for row, sample_index, weight in zip(probabilities, sample_mapping, weights):
        totals[sample_index] += weight
        windows[sample_index] += 1
        for label in ["positive", "negative", "neutral"]:
            sums[sample_index][label] += float(row[label_indices[label]]) * weight

    results: list[SectionScore] = []
    for sample_index in range(sample_count):
        if totals[sample_index] == 0:
            raise ValueError(f"sample {sample_index} produced no token windows")
        positive = sums[sample_index]["positive"] / totals[sample_index]
        negative = sums[sample_index]["negative"] / totals[sample_index]
        neutral = sums[sample_index]["neutral"] / totals[sample_index]
        results.append(
            SectionScore(
                positive=positive,
                negative=negative,
                neutral=neutral,
                tone=positive - negative,
                windows=windows[sample_index],
                tokens=totals[sample_index],
            )
        )
    return results


class FinBertWindowScorer:
    def __init__(
        self,
        config: ScoreConfig,
        cache_dir: Path,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        self.window_batch_size = config.window_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            cache_dir=cache_dir,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            cache_dir=cache_dir,
            use_safetensors=True,
        )
        self.model.eval().to(self.device)

        labels = {
            str(label).lower(): int(index)
            for index, label in self.model.config.id2label.items()
        }
        required = {"positive", "negative", "neutral"}
        if set(labels) != required:
            raise ValueError(f"Unexpected FinBERT labels: {labels}")
        self.label_indices = labels

        torch.manual_seed(0)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return self.torch.cuda.get_device_name(self.device)
        return "CPU"

    def score_texts(self, texts: Sequence[str]) -> list[SectionScore]:
        if not texts or any(not str(text).strip() for text in texts):
            raise ValueError("score_texts requires non-empty text inputs")

        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.config.max_length,
            stride=self.config.stride,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
            padding=False,
        )
        mapping = [int(value) for value in encoded["overflow_to_sample_mapping"]]
        content_lengths = [
            len(input_ids) - sum(special_mask)
            for input_ids, special_mask in zip(
                encoded["input_ids"], encoded["special_tokens_mask"]
            )
        ]

        probabilities: list[list[float]] = []
        model_keys = [
            key
            for key in ["input_ids", "attention_mask", "token_type_ids"]
            if key in encoded
        ]
        for start in range(0, len(mapping), self.window_batch_size):
            end = min(start + self.window_batch_size, len(mapping))
            features = [
                {key: encoded[key][index] for key in model_keys}
                for index in range(start, end)
            ]
            batch = self.tokenizer.pad(
                features,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.inference_mode():
                if self.device.type == "cuda":
                    with self.torch.autocast(
                        device_type="cuda", dtype=self.torch.float16
                    ):
                        logits = self.model(**batch).logits
                else:
                    logits = self.model(**batch).logits
                chunk_probabilities = self.torch.softmax(logits.float(), dim=-1)
            probabilities.extend(chunk_probabilities.cpu().tolist())

        return aggregate_window_probabilities(
            probabilities,
            mapping,
            content_lengths,
            len(texts),
            self.config.stride,
            self.label_indices,
        )
