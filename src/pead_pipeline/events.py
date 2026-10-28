from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MASTER_REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "Tone_Pres",
    "Tone_QA",
    "CAR_5",
    "CAR_10",
    "CAR_60",
    "SUE_Proxy",
    "Size",
    "VOL",
    "Turnover",
]

MASTER_VALUE_COLUMNS = [
    "Tone_Pres",
    "Tone_QA",
    "CAR_5",
    "CAR_10",
    "CAR_60",
    "SUE_Proxy",
    "Size",
    "VOL",
    "Turnover",
]

NLP_REQUIRED_COLUMNS = ["ticker", "date", "Tone_Score"]

_EMBEDDED_TIMESTAMP = re.compile(
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4},?\s+"
    r"\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|AM|PM)(?:\s+(?:ET|EST|EDT))?",
    flags=re.IGNORECASE,
)


def normalize_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = re.sub(r"\s+", "", str(value)).upper()
    # The source scraper left an unmatched closing parenthesis on 31 rows
    # (for example, CIB)). Parentheses are not valid ticker characters.
    return re.sub(r"\)+$", "", normalized)


def _normalize_date_text(value: object) -> object:
    if value is None or value is pd.NA:
        return pd.NA
    if not isinstance(value, (list, tuple)) and pd.isna(value):
        return pd.NA
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 1
        and isinstance(value[-1], str)
    ):
        text = value[-1].strip()
    else:
        text = str(value).strip()
    # A legacy scraper stored a small subset as the repr of
    # [earnings-call title, timestamp]. Preserve the title in the source CSV,
    # but parse the timestamp element for the event key.
    if text.startswith("["):
        try:
            legacy_parts = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            legacy_parts = None
        if (
            isinstance(legacy_parts, (list, tuple))
            and len(legacy_parts) >= 1
            and isinstance(legacy_parts[-1], str)
        ):
            text = legacy_parts[-1].strip()
    embedded_timestamps = _EMBEDDED_TIMESTAMP.findall(text)
    if embedded_timestamps:
        text = embedded_timestamps[-1]
    text = re.sub(r"(?i)\ba\.m\.", "AM", text)
    text = re.sub(r"(?i)\bp\.m\.", "PM", text)
    text = re.sub(r"(?i)\s+(ET|EST|EDT)\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_earnings_timestamp(values: pd.Series) -> pd.Series:
    """Parse source timestamps and attach the America/New_York timezone."""

    cleaned = values.map(_normalize_date_text)
    parsed = pd.to_datetime(cleaned, format="mixed", errors="coerce")
    return parsed.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    )


def build_event_id(ticker: str, timestamp: object, source_row_id: int | None = None) -> str:
    normalized_ticker = normalize_ticker(ticker)
    if pd.isna(timestamp):
        basis = f"unparsed|{normalized_ticker}|{source_row_id}"
        prefix = "unparsed"
    else:
        ts = pd.Timestamp(timestamp)
        basis = f"{normalized_ticker}|{ts.isoformat()}"
        prefix = "event"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def classify_market_session(timestamp: object) -> str:
    if pd.isna(timestamp):
        return "unknown"
    local_time = pd.Timestamp(timestamp).time()
    if local_time < pd.Timestamp("09:30").time():
        return "pre_market"
    if local_time < pd.Timestamp("16:00").time():
        return "regular_hours"
    return "post_market"


def target_start_rule(session: str) -> str:
    if session == "pre_market":
        return "same_trading_day_after_call"
    if session in {"regular_hours", "post_market"}:
        return "next_trading_day"
    return "manual_review"


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def audit_master(
    frame: pd.DataFrame, reference_timestamps: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(frame, MASTER_REQUIRED_COLUMNS, "master dataset")
    audited = frame.copy()
    audited.insert(0, "source_row_id", range(1, len(audited) + 1))
    audited["ticker_normalized"] = audited["ticker"].map(normalize_ticker)
    audited["source_timestamp_et"] = parse_earnings_timestamp(audited["date"])
    audited["event_timestamp_et"] = audited["source_timestamp_et"]
    audited["timestamp_resolution"] = "source_timestamp"
    audited["timestamp_recovery_candidate_count"] = 0

    midnight_mask = (
        audited["source_timestamp_et"].notna()
        & audited["source_timestamp_et"].dt.hour.eq(0)
        & audited["source_timestamp_et"].dt.minute.eq(0)
        & audited["source_timestamp_et"].dt.second.eq(0)
    )
    if reference_timestamps is not None and midnight_mask.any():
        _require_columns(
            reference_timestamps,
            ["ticker_normalized", "event_timestamp_et"],
            "reference timestamp dataset",
        )
        reference = reference_timestamps[
            reference_timestamps["event_timestamp_et"].notna()
        ].copy()
        reference = reference[
            reference["event_timestamp_et"].dt.hour.ne(0)
            | reference["event_timestamp_et"].dt.minute.ne(0)
        ]
        reference["calendar_date"] = reference["event_timestamp_et"].dt.date
        grouped = reference.groupby(
            ["ticker_normalized", "calendar_date"], dropna=False
        )["event_timestamp_et"]
        candidate_counts = grouped.nunique()
        unique_candidates = grouped.first()[candidate_counts.eq(1)].to_dict()
        count_lookup = candidate_counts.to_dict()

        for index in audited.index[midnight_mask]:
            key = (
                audited.at[index, "ticker_normalized"],
                audited.at[index, "source_timestamp_et"].date(),
            )
            count = int(count_lookup.get(key, 0))
            audited.at[index, "timestamp_recovery_candidate_count"] = count
            if count == 1:
                audited.at[index, "event_timestamp_et"] = unique_candidates[key]
                audited.at[index, "timestamp_resolution"] = (
                    "recovered_from_nlp_scores"
                )
            else:
                audited.at[index, "event_timestamp_et"] = pd.NaT
                audited.at[index, "timestamp_resolution"] = (
                    "unresolved_no_candidate" if count == 0 else "unresolved_ambiguous"
                )
    elif midnight_mask.any():
        audited.loc[midnight_mask, "timestamp_resolution"] = (
            "source_midnight_unverified"
        )
    audited["event_id"] = [
        build_event_id(ticker, timestamp, row_id)
        for ticker, timestamp, row_id in zip(
            audited["ticker_normalized"],
            audited["event_timestamp_et"],
            audited["source_row_id"],
        )
    ]
    audited["market_session"] = audited["event_timestamp_et"].map(
        classify_market_session
    )
    audited["target_start_rule"] = audited["market_session"].map(target_start_rule)

    group_size = audited.groupby("event_id", dropna=False)["event_id"].transform("size")
    distinct_values = audited.groupby("event_id", dropna=False)[MASTER_VALUE_COLUMNS].transform(
        lambda column: column.nunique(dropna=False)
    )
    max_distinct = distinct_values.max(axis=1)
    audited["event_row_count"] = group_size.astype(int)
    audited["event_distinct_value_count"] = max_distinct.astype(int)
    audited["event_status"] = "unique"
    audited.loc[(group_size > 1) & (max_distinct == 1), "event_status"] = (
        "exact_duplicate"
    )
    audited.loc[(group_size > 1) & (max_distinct > 1), "event_status"] = "conflict"
    audited.loc[audited["event_timestamp_et"].isna(), "event_status"] = (
        "unresolved_timestamp"
    )

    safe = (
        audited[audited["event_status"].isin(["unique", "exact_duplicate"])]
        .sort_values("source_row_id")
        .drop_duplicates("event_id", keep="first")
        .copy()
    )
    conflicts = audited[audited["event_status"] == "conflict"].copy()
    return audited, safe, conflicts


def audit_nlp_scores(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, NLP_REQUIRED_COLUMNS, "NLP score dataset")
    audited = frame.copy()
    audited.insert(0, "source_row_id", range(1, len(audited) + 1))
    audited["ticker_normalized"] = audited["ticker"].map(normalize_ticker)
    audited["event_timestamp_et"] = parse_earnings_timestamp(audited["date"])
    audited["event_id"] = [
        build_event_id(ticker, timestamp, row_id)
        for ticker, timestamp, row_id in zip(
            audited["ticker_normalized"],
            audited["event_timestamp_et"],
            audited["source_row_id"],
        )
    ]
    audited["exact_row_duplicate"] = frame.duplicated(keep=False)
    audited["event_row_count"] = audited.groupby("event_id")["event_id"].transform(
        "size"
    )
    audited["tone_score_distinct_count"] = audited.groupby("event_id")[
        "Tone_Score"
    ].transform(lambda column: column.nunique(dropna=False))
    return audited


def serialize_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["source_timestamp_et", "event_timestamp_et"]:
        if column in result:
            result[column] = result[column].map(
                lambda value: "" if pd.isna(value) else pd.Timestamp(value).isoformat()
            )
    return result


@dataclass(frozen=True)
class RawTranscriptStatus:
    exists: bool
    path: str
    size_bytes: int | None
    rows: int | None
    columns: list[str] | None
    valid_schema: bool | None
    error: str | None


def inspect_raw_transcripts(path: Path) -> RawTranscriptStatus:
    if not path.exists():
        return RawTranscriptStatus(False, str(path), None, None, None, None, None)
    try:
        raw = pd.read_pickle(path)
        if not isinstance(raw, pd.DataFrame):
            raw = pd.DataFrame(raw)
        columns = [str(column) for column in raw.columns]
        required = {"ticker", "date", "transcript"}
        return RawTranscriptStatus(
            True,
            str(path),
            path.stat().st_size,
            len(raw),
            columns,
            required.issubset(columns),
            None,
        )
    except Exception as exc:  # pragma: no cover - only exercised with external files
        return RawTranscriptStatus(
            True, str(path), path.stat().st_size, None, None, False, str(exc)
        )

