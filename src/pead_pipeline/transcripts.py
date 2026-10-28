from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .events import build_event_id, normalize_ticker, parse_earnings_timestamp


RAW_REQUIRED_COLUMNS = ["date", "exchange", "q", "ticker", "transcript"]

PREPARED_MARKER = re.compile(r"Prepared Remarks\s*:", re.IGNORECASE)
QA_MARKER = re.compile(
    r"Questions\s*(?:&|and)\s*Answers\s*:", re.IGNORECASE
)
FOOTER_MARKER = re.compile(
    r"\n(?:Duration\s*:|Call participants\s*:|More [^\n]{0,40} analysis\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
WORD = re.compile(r"\S+")
MIN_SECTION_WORDS = 20


@dataclass(frozen=True)
class TranscriptSplit:
    status: str
    prepared_start: int | None
    prepared_end: int | None
    qa_start: int | None
    qa_end: int | None
    footer_start: int | None


def _require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"raw transcript dataset is missing columns: {missing}")


def _raw_date_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def split_transcript(text: str) -> TranscriptSplit:
    prepared = PREPARED_MARKER.search(text)
    if prepared is None:
        return TranscriptSplit(
            "prepared_marker_missing", None, None, None, None, None
        )

    qa = QA_MARKER.search(text, prepared.end())
    if qa is None:
        return TranscriptSplit(
            "qa_marker_missing", prepared.end(), None, None, None, None
        )

    footer = FOOTER_MARKER.search(text, qa.end())
    if footer is None:
        return TranscriptSplit(
            "footer_marker_missing",
            prepared.end(),
            qa.start(),
            qa.end(),
            None,
            None,
        )

    prepared_text = text[prepared.end() : qa.start()].strip()
    qa_text = text[qa.end() : footer.start()].strip()
    if not prepared_text:
        status = "prepared_text_empty"
    elif not qa_text:
        status = "qa_text_empty"
    elif len(WORD.findall(prepared_text)) < MIN_SECTION_WORDS:
        status = "prepared_text_too_short"
    elif len(WORD.findall(qa_text)) < MIN_SECTION_WORDS:
        status = "qa_text_too_short"
    else:
        status = "split_complete"
    return TranscriptSplit(
        status,
        prepared.end(),
        qa.start(),
        qa.end(),
        footer.start(),
        footer.start(),
    )


def _slice(text: str, start: object, end: object) -> str:
    if pd.isna(start) or pd.isna(end):
        return ""
    return text[int(start) : int(end)].strip()


def build_transcript_audit(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(raw, RAW_REQUIRED_COLUMNS)

    rows: list[dict[str, object]] = []
    timestamps = parse_earnings_timestamp(raw["date"])
    for position, ((_, source), timestamp) in enumerate(
        zip(raw.iterrows(), timestamps), start=1
    ):
        ticker = normalize_ticker(source["ticker"])
        text = str(source["transcript"])
        split = split_transcript(text)
        prepared_text = _slice(text, split.prepared_start, split.prepared_end)
        qa_text = _slice(text, split.qa_start, split.qa_end)
        rows.append(
            {
                "source_row_id": position,
                "source_position": position - 1,
                "ticker": source["ticker"],
                "ticker_normalized": ticker,
                "date_raw": _raw_date_text(source["date"]),
                "event_timestamp_et": timestamp,
                "event_id": build_event_id(ticker, timestamp, position),
                "exchange": source["exchange"],
                "quarter": source["q"],
                "transcript_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "transcript_char_count": len(text),
                "transcript_word_count": len(WORD.findall(text)),
                "split_status": split.status,
                "prepared_start": split.prepared_start,
                "prepared_end": split.prepared_end,
                "qa_start": split.qa_start,
                "qa_end": split.qa_end,
                "footer_start": split.footer_start,
                "prepared_char_count": len(prepared_text),
                "prepared_word_count": len(WORD.findall(prepared_text)),
                "qa_char_count": len(qa_text),
                "qa_word_count": len(WORD.findall(qa_text)),
            }
        )

    audited = pd.DataFrame(rows)
    grouped = audited.groupby("event_id", dropna=False)
    audited["event_row_count"] = grouped["event_id"].transform("size")
    audited["event_distinct_transcript_count"] = grouped[
        "transcript_sha256"
    ].transform("nunique")
    metadata_distinct = grouped[["exchange", "quarter"]].transform(
        lambda column: column.nunique(dropna=False)
    )
    audited["event_distinct_metadata_count"] = metadata_distinct.max(axis=1)

    audited["event_status"] = "unique"
    duplicate = audited["event_row_count"] > 1
    exact = (
        duplicate
        & audited["event_distinct_transcript_count"].eq(1)
        & audited["event_distinct_metadata_count"].eq(1)
    )
    conflict = duplicate & ~exact
    audited.loc[exact, "event_status"] = "exact_duplicate"
    audited.loc[conflict, "event_status"] = "conflict"
    audited.loc[audited["event_timestamp_et"].isna(), "event_status"] = (
        "unresolved_timestamp"
    )

    safe = (
        audited[
            audited["event_status"].isin(["unique", "exact_duplicate"])
            & audited["split_status"].eq("split_complete")
        ]
        .sort_values("source_row_id")
        .drop_duplicates("event_id", keep="first")
        .copy()
    )
    review = audited[
        ~audited["source_row_id"].isin(safe["source_row_id"])
    ].copy()
    return audited, safe, review


def write_section_corpus(
    raw: pd.DataFrame,
    safe: pd.DataFrame,
    path: Path,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    written = 0
    try:
        with temporary.open("wb") as binary_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=4,
                fileobj=binary_handle,
                mtime=0,
            ) as compressed_handle:
                with io.TextIOWrapper(
                    compressed_handle, encoding="utf-8", newline="\n"
                ) as handle:
                    for _, audit_row in safe.iterrows():
                        source = raw.iloc[int(audit_row["source_position"])]
                        text = str(source["transcript"])
                        record = {
                            "event_id": audit_row["event_id"],
                            "ticker": audit_row["ticker_normalized"],
                            "event_timestamp_et": pd.Timestamp(
                                audit_row["event_timestamp_et"]
                            ).isoformat(),
                            "date_raw": audit_row["date_raw"],
                            "exchange": audit_row["exchange"],
                            "quarter": audit_row["quarter"],
                            "transcript_sha256": audit_row["transcript_sha256"],
                            "prepared_text": _slice(
                                text,
                                audit_row["prepared_start"],
                                audit_row["prepared_end"],
                            ),
                            "qa_text": _slice(
                                text, audit_row["qa_start"], audit_row["qa_end"]
                            ),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return written


def serialize_audit_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["event_timestamp_et"] = result["event_timestamp_et"].map(
        lambda value: "" if pd.isna(value) else pd.Timestamp(value).isoformat()
    )
    return result
