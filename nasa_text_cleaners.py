"""
NASA text parsing and cleaning helpers.

This module has two parsers:
1. nasa_report_to_clean_df: NASA report / flight-plan style OCR text.
2. nasa_transcripts_to_clean_df: NASA transcript / mission-commentary OCR text.

Both return a DataFrame with: id, document, metadata.
Use build_debug_df(df) for inspection. The output DataFrames can be chunked later.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Shared helpers
# =============================================================================

def validate_cleaned_df(df: pd.DataFrame) -> None:
    required = {"id", "document", "metadata"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["id"].isna().any():
        raise ValueError("Some ids are missing.")
    if df["document"].isna().any():
        raise ValueError("Some documents are missing.")
    if not df["metadata"].map(lambda x: isinstance(x, dict)).all():
        raise ValueError("Every metadata value must be a dict.")
    duplicate_count = df["id"].astype(str).duplicated().sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate ids.")


def to_clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert the cleaned DataFrame to a list of Python records.

    This is optional. Most of the time you can keep using the DataFrame.
    This function does not prepare a ChromaDB payload.
    """
    validate_cleaned_df(df)
    return df.to_dict(orient="records")


def build_debug_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten metadata for inspection only. Do not pass debug_df to ChromaDB."""
    validate_cleaned_df(df)
    return pd.concat(
        [
            df[["id", "document"]].reset_index(drop=True),
            pd.json_normalize(df["metadata"]).reset_index(drop=True),
        ],
        axis=1,
    )


def _clean_text(s: str) -> str:
    s = re.sub(r"([A-Za-z]+)-\s*\n\s*([a-z]+)", r"\1\2", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# =============================================================================
# Report / flight-plan parser
# =============================================================================

def nasa_report_to_clean_df(
    raw_text: str | None = None,
    file_path: str | Path | None = None,
    start_page: int = 1,
    stop_heading: str | None = None,
    mission: str = "",
    source_file: str | None = None,
    collection: str = "",
    report_type: str = "",
    drop_page_keywords: list[str] | None = None,
    drop_pages_with_figures: bool = True,
    drop_pages_with_tables: bool = True,
    include_section_in_document: bool = True,
) -> pd.DataFrame:
    """
    Clean NASA report / flight-plan OCR text and return a Chroma-ready DataFrame.

    Shared metadata keys:
        collection, mission, source_type, source_file, source_path, doc_id

    Report-specific metadata keys:
        report_type, section_path, section_depth, top_section_id,
        top_section_title, page_start, page_end
    """
    if raw_text is not None and isinstance(raw_text, str) and not raw_text.strip():
        raw_text = None
    if file_path is not None and isinstance(file_path, str) and not file_path.strip():
        file_path = None
    if raw_text is None and file_path is None:
        raise ValueError("Provide either raw_text or file_path.")
    if raw_text is not None and file_path is not None:
        raise ValueError("Provide only one of raw_text or file_path, not both.")

    if file_path is not None:
        path = Path(file_path)
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        source_file = source_file or path.name
        doc_id = path.stem
        source_path = str(path)
    else:
        source_file = source_file or "raw_text"
        doc_id = Path(source_file).stem
        source_path = ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?m)^---\s*PAGE\s*(\d+)\s*---\s*$", text)

    pages: list[tuple[int, str]] = []
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            try:
                pages.append((int(parts[i]), parts[i + 1]))
            except ValueError:
                continue
    else:
        pages = [(1, text)]

    figure_caption_re = re.compile(
        r"(?mi)^\s*(?:FIG|FIGURE)\.?\s+[A-Z0-9]+(?:[-.][A-Z0-9]+)*\b.*$"
    )
    table_caption_re = re.compile(
        r"(?mi)^\s*TABLE\s+[A-Z0-9]+(?:[-.][A-Z0-9]+)*\b.*$"
    )

    def page_has_drop_keyword(page: str, keywords: list[str] | None) -> bool:
        if not keywords:
            return False
        normalized_page = re.sub(r"\s+", " ", page).upper()
        lines = [re.sub(r"\s+", " ", line.strip()).upper() for line in page.splitlines() if line.strip()]
        for keyword in keywords:
            keyword_norm = re.sub(r"\s+", " ", keyword.strip()).upper()
            if not keyword_norm:
                continue
            if keyword_norm in {"FIG", "FIG."}:
                if any(re.match(r"^FIG(?:\.|URE)?\b", line) for line in lines):
                    return True
                continue
            if len(keyword_norm) <= 5 and " " not in keyword_norm:
                if re.search(rf"\b{re.escape(keyword_norm)}\b", normalized_page):
                    return True
                continue
            if keyword_norm in normalized_page:
                return True
        return False

    def strip_page_noise(page: str) -> str:
        page = page.replace("\r\n", "\n").replace("\r", "\n")
        page = re.sub(r"(?m)\A\s*\d+\s*\n+", "", page)
        page = re.sub(r"(?m)\n+\s*\d+\s*\Z", "", page)
        page = re.sub(r"(?m)^\s*[A-Z]?\d+-\d+(?:/[A-Z]?\d+-\d+)?\s*$", "", page)
        page = re.sub(
            r"(?mi)^\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii|xxiv|xxv|xxvi|xxvii|xxviii)\s*$",
            "",
            page,
        )
        common_header_patterns = [
            r"NASA\s+SP-\d+",
            r"APOLLO\s+\d+\s+MISSION\s+REPORT",
            r"NATIONAL AERONAUTICS AND SPACE ADMINISTRATION",
            r"MANNED SPACECRAFT CENTER",
            r"HOUSTON,?\s*TEXAS",
            r"GEORGE C\.?\s*MARSHALL SPACE FLIGHT CENTER",
            r"MPR[-– ]*SAT[-– ]*FE[-– ]*\d+[-– ]*\d+",
            r"SATURN V LAUNCH VEHICLE FLIGHT EVALUATION REPORT.*",
            r"APOLLO\s+\d+\s+MISSION",
            r"APOLLO\s+\d+\s+FLIGHT PLAN",
            r"AS-\d+/CSM-\d+/LM-\d+",
            r"Revision A",
        ]
        for pattern in common_header_patterns:
            page = re.sub(rf"(?mi)^\s*{pattern}\s*$", "", page)
        page = re.sub(r"(?m)^\s*#{2,}\s*$", "", page)
        return page.strip()

    def looks_like_visual_or_table_body(page: str) -> bool:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        if not lines:
            return True
        if len(lines) < 8:
            return False
        prose_lines = short_or_numeric_lines = numericish_lines = 0
        for line in lines:
            alpha_count = sum(ch.isalpha() for ch in line)
            digit_count = sum(ch.isdigit() for ch in line)
            punct_count = sum(not ch.isalnum() and not ch.isspace() for ch in line)
            words = re.findall(r"[A-Za-z]{2,}", line)
            if len(line) <= 28 or digit_count > alpha_count:
                short_or_numeric_lines += 1
            if digit_count + punct_count > alpha_count:
                numericish_lines += 1
            if len(line) >= 55 and len(words) >= 7:
                prose_lines += 1
        return prose_lines < 2 and (
            short_or_numeric_lines / len(lines) > 0.72 or numericish_lines / len(lines) > 0.55
        )

    kept_pages: list[tuple[int, str]] = []
    for page_num, page in pages:
        if page_num < start_page:
            continue
        if stop_heading and re.search(rf"(?mi)^\s*{re.escape(stop_heading)}\s*$", page):
            break
        page = strip_page_noise(page)
        if not page:
            continue
        if drop_pages_with_figures and figure_caption_re.search(page):
            continue
        if drop_pages_with_tables and table_caption_re.search(page):
            continue
        if page_has_drop_keyword(page, drop_page_keywords):
            continue
        if looks_like_visual_or_table_body(page):
            continue
        kept_pages.append((page_num, page))

    line_items: list[tuple[int, str]] = []
    for page_num, page in kept_pages:
        page = figure_caption_re.sub("", page)
        page = table_caption_re.sub("", page)
        for line in page.splitlines():
            line_items.append((page_num, line))
        line_items.append((page_num, ""))

    def roman_to_int(s: str) -> int | None:
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
        total = previous = 0
        for char in reversed(s.upper()):
            value = values.get(char, 0)
            total = total - value if value < previous else total + value
            previous = max(previous, value)
        return total or None

    def parse_heading(line: str) -> dict[str, Any] | None:
        s = line.strip()
        if not s or len(s) > 140:
            return None
        if re.match(r"^\([A-Za-z0-9]+\)\s+", s):
            return None
        m = re.match(r"(?i)^SECTION\s+([IVXLCDM]+|\d+)\s*(?:[-–]+\s*(.+))?$", s)
        if m:
            raw_id = m.group(1).upper()
            section_id = str(roman_to_int(raw_id)) if raw_id.isalpha() else raw_id
            title = (m.group(2) or "").strip()
            label = f"{section_id}. {title}" if title else f"SECTION {section_id}"
            return {"id": section_id, "title": title, "label": label, "numbered": True, "pending": not bool(title)}
        m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", s)
        if m:
            section_id, title = m.group(1), m.group(2).strip()
            if not re.search(r"[A-Za-z]", title):
                return None
            letters = re.findall(r"[A-Za-z]", title)
            upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
            if upper_ratio < 0.60 or len(title) > 130:
                return None
            return {"id": section_id, "title": title, "label": f"{section_id}. {title}", "numbered": True, "pending": False}
        if re.search(r"[.;,]$", s):
            return None
        words = re.findall(r"[A-Za-z][A-Za-z'/-]*", s)
        if not (1 <= len(words) <= 10):
            return None
        letters = re.findall(r"[A-Za-z]", s)
        if not letters:
            return None
        first_alpha = next((ch for ch in s if ch.isalpha()), "")
        if first_alpha and not first_alpha.isupper():
            return None
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        cap_words = sum(word[0].isupper() for word in words)
        if upper_ratio >= 0.75 or cap_words >= max(1, len(words) - 1):
            bad_headings = {"TABLE OF CONTENTS", "LIST OF ILLUSTRATIONS", "LIST OF TABLES", "ABBREVIATIONS", "SYMBOLS AND ABBREVIATIONS"}
            if s.upper() in bad_headings:
                return None
            return {"id": "", "title": s, "label": s, "numbered": False, "pending": False}
        return None

    def is_heading(line: str) -> bool:
        return parse_heading(line) is not None

    def clean_block(s: str) -> str:
        s = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\b[A-Z]?\d+-\d+(?:/[A-Z]?\d+-\d+)?\b$", "", s).strip()
        return s

    def ends_mid_sentence(block: str) -> bool:
        return not bool(re.search(r"[.!?;:)\"']\s*$", block.rstrip()))

    def starts_like_continuation(block: str) -> bool:
        return bool(re.match(r"^(?:[a-z]|and\b|or\b|but\b|because\b|which\b|that\b|of\b|in\b|on\b|for\b|with\b|by\b|as\b|from\b|to\b|the\b|this\b)", block.lstrip()))

    blocks: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    buffer_pages: list[int] = []
    for page_num, raw_line in line_items:
        line = raw_line.strip()
        if not line:
            if buffer:
                blocks.append((" ".join(buffer), min(buffer_pages), max(buffer_pages)))
                buffer, buffer_pages = [], []
            continue
        if re.fullmatch(r"[A-Z]?\d+-\d+(?:/[A-Z]?\d+-\d+)?", line) or re.fullmatch(r"\d+", line):
            continue
        if is_heading(line):
            if buffer:
                blocks.append((" ".join(buffer), min(buffer_pages), max(buffer_pages)))
                buffer, buffer_pages = [], []
            blocks.append((line, page_num, page_num))
        else:
            buffer.append(line)
            buffer_pages.append(page_num)
    if buffer:
        blocks.append((" ".join(buffer), min(buffer_pages), max(buffer_pages)))
    blocks = [(clean_block(block), page_start, page_end) for block, page_start, page_end in blocks if clean_block(block)]

    merged_blocks: list[tuple[str, int, int]] = []
    for block, page_start, page_end in blocks:
        if (
            merged_blocks
            and not is_heading(merged_blocks[-1][0])
            and not is_heading(block)
            and ends_mid_sentence(merged_blocks[-1][0])
            and starts_like_continuation(block)
        ):
            previous_block, previous_start, previous_end = merged_blocks[-1]
            merged_blocks[-1] = (clean_block(previous_block + " " + block), min(previous_start, page_start), max(previous_end, page_end))
        else:
            merged_blocks.append((block, page_start, page_end))

    rows: list[dict[str, Any]] = []
    section_stack: list[dict[str, Any]] = []
    pending_section: dict[str, Any] | None = None
    for block, page_start, page_end in merged_blocks:
        heading = parse_heading(block)
        if heading:
            if heading.get("pending"):
                pending_section = heading
                continue
            if pending_section and not heading["numbered"]:
                section_id = pending_section["id"]
                title = heading["title"]
                section_stack = [{"id": section_id, "title": title, "label": f"{section_id}. {title}", "numbered": True}]
                pending_section = None
                continue
            pending_section = None
            if heading["numbered"]:
                depth = heading["id"].count(".") + 1 if heading["id"] else 1
                section_stack = section_stack[: depth - 1]
                section_stack.append(heading)
            else:
                numbered_depth = 0
                for i, item in enumerate(section_stack):
                    if item["id"]:
                        numbered_depth = i + 1
                if numbered_depth:
                    target_depth = numbered_depth + 1
                    section_stack = section_stack[: target_depth - 1]
                    section_stack.append(heading)
                else:
                    section_stack = [heading]
            continue

        pending_section = None
        section_path = " > ".join(item["label"] for item in section_stack)
        top_numbered = next((item for item in section_stack if item["id"]), None)
        document = f"{section_path}\n\n{block}".strip() if include_section_in_document and section_path else block
        metadata = {
            "collection": collection,
            "mission": mission,
            "source_type": "report",
            "source_file": source_file or "",
            "source_path": source_path,
            "doc_id": doc_id,
            "report_type": report_type,
            "section_path": section_path,
            "section_depth": len(section_stack),
            "top_section_id": top_numbered["id"] if top_numbered else "",
            "top_section_title": top_numbered["title"] if top_numbered else "",
            "page_start": int(page_start),
            "page_end": int(page_end),
        }
        rows.append({"id": f"{doc_id}_report_{len(rows):05d}", "document": document, "metadata": metadata})

    return pd.DataFrame(rows, columns=["id", "document", "metadata"])


# =============================================================================
# Transcript / commentary parser
# =============================================================================

def nasa_transcripts_to_clean_df(
    input_path: str | Path | None = None,
    raw_text: str | None = None,
    file_pattern: str = "*.txt",
    parser: str = "auto",
    skiprows: int = 0,
    start_page: int | None = None,
    mission: str = "",
    collection: str = "",
    source_file: str = "",
    implicit_speaker: str = "PAO",
    speakers: list[str] | None = None,
) -> pd.DataFrame:
    """Parse NASA transcript txt files into a Chroma-ready DataFrame."""
    if raw_text is not None and isinstance(raw_text, str) and not raw_text.strip():
        raw_text = None
    if input_path is not None and isinstance(input_path, str) and not input_path.strip():
        input_path = None
    if raw_text is None and input_path is None:
        raise ValueError("Provide either raw_text or input_path.")
    if raw_text is not None and input_path is not None:
        raise ValueError("Provide only one of raw_text or input_path, not both.")
    if parser not in {"auto", "challenger", "apollo", "pao"}:
        raise ValueError("parser must be one of: auto, challenger, apollo, pao")
    input_path = Path(input_path) if input_path is not None else None

    def colon_timestamp_to_seconds(ts: str) -> int | None:
        try:
            parts = [int(x) for x in ts.split(":")]
        except ValueError:
            return None
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        return None

    def apollo_timestamp_to_seconds(ts: str | None) -> int | None:
        if not ts:
            return None
        parts = ts.split()
        if len(parts) != 4:
            return None
        try:
            d, h, m, s = [int(x) for x in parts]
        except ValueError:
            return None
        return ((d * 24 + h) * 60 + m) * 60 + s

    def cut_to_start_page(text: str) -> str:
        if start_page is None:
            return text
        page_re = re.compile(rf"(?mi)^---\s*PAGE\s*{start_page}\s*---\s*$")
        m = page_re.search(text)
        return text[m.start():] if m else text

    def infer_doc_metadata(path: Path | None) -> dict[str, str]:
        if path is None:
            src_file = source_file or "raw_text"
            src_path = ""
            doc_id = Path(src_file).stem
            parent_collection = collection
        else:
            src_file = path.name
            src_path = str(path)
            doc_id = path.stem
            parent_collection = collection or path.parent.name
        doc_number_match = re.match(r"^(\d+)-", src_file)
        doc_number = doc_number_match.group(1) if doc_number_match else ""
        sts_match = re.search(r"(STS-\d+[A-Z]?)", src_file)
        inferred_mission = sts_match.group(1) if sts_match else ""
        return {
            "collection": parent_collection or "",
            "source_file": src_file or "",
            "source_path": src_path or "",
            "doc_id": doc_id or "",
            "doc_number": doc_number,
            "mission": mission or inferred_mission or "",
        }

    def make_metadata(meta_base: dict[str, str], transcript_format: str, utterance_index: int, speaker: str = "", timestamp: str = "", timestamp_sec: int | None = None, speaker_id: int | None = None, segment_marker: str = "") -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "collection": meta_base["collection"],
            "mission": meta_base["mission"],
            "source_type": "transcript",
            "source_file": meta_base["source_file"],
            "source_path": meta_base["source_path"],
            "doc_id": meta_base["doc_id"],
            "transcript_format": transcript_format,
            "utterance_index": int(utterance_index),
        }
        if meta_base.get("doc_number"):
            metadata["doc_number"] = meta_base["doc_number"]
        if speaker:
            metadata["speaker"] = speaker
        if speaker_id is not None:
            metadata["speaker_id"] = int(speaker_id)
        if timestamp:
            metadata["timestamp"] = timestamp
        if timestamp_sec is not None:
            metadata["timestamp_sec"] = int(timestamp_sec)
        if segment_marker:
            metadata["segment_marker"] = segment_marker
        return metadata

    def make_row(row_id: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"id": row_id, "document": _clean_text(text), "metadata": metadata}

    def normalize_speaker(s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip().upper())
        return {"P AO": "PAO", "P A O": "PAO", "CAP COM": "CAPCOM", "CAPCOMM": "CAPCOM"}.get(s, s)

    def build_speaker_pattern(extra_speakers: list[str] | None = None) -> re.Pattern[str]:
        base_speakers = [
            "PAO", "P AO", "P A O", "MCC", "MSC", "KSC", "LCC", "CAPCOM", "CAP COM", "COMM TECH",
            "CDR", "CMP", "LMP", "CDR/LMP", "CDR-CMP", "CMP/LMP", "LMP/CDR", "CMP/CDR",
            "CDR-EVA", "LMP-EVA", "CDR-LM", "LMP-LM", "SC", "MS", "CC", "CT", "MSFN", "F",
            "HORNET", "R", "R-1", "R-2", "R-3", "AB", "IWO", "P-1", "P-2", "CMF", "LMF", "IMP",
        ]
        if extra_speakers:
            base_speakers.extend(extra_speakers)
        base_speakers = sorted(set(base_speakers), key=len, reverse=True)
        return re.compile(r"^\s*(" + "|".join(re.escape(x) for x in base_speakers) + r")\s*$", flags=re.IGNORECASE)

    speaker_line_re = build_speaker_pattern(speakers)

    def remove_common_artifact_lines(text: str, remove_numeric_only_lines: bool) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        artifact_patterns = [
            r"^\s*---\s*PAGE\s*\d+\s*---\s*$", r"^\s*#{2,}\s*$",
            r"^\s*(CONFIDENTIAL|CONFIDENTIA|CONFIDENTI|UNCLASSIFIED)\s*$",
            r"^\s*NATIONAL AERONAUTICS AND SPACE ADMINISTRATION\s*$", r"^\s*MANNED SPACECRAFT CENTER\s*$", r"^\s*HOUSTON,?\s*TEXAS\s*$",
            r"^\s*APOLLO\s+\d+\s*-?\s*AIR-TO-GROUND\s+VOICE\s+TRANSCRIPTION\s*$",
            r"^\s*APOLLO\s+\d+\s*-?\s*ONBOARD\s+VOICE\s+TRANSCRIPTION\s*$",
            r"^\s*APOLLO\s+\d+\s*-?\s*MISSION\s+COMMENTARY\s*$",
            r"^\s*APOLLO\s+\d+\s+MISSION\s+COMMENTARY.*$", r"^\s*APOLLO\s+\d+\s+STATUS\s+REPORT.*$", r"^\s*APOLLO\s+\d+\s+SPACECRAFT\s+COMMENTARY.*$",
            r"^\s*Tape\s*\d+/\d+\s*-?\s*Page\s*\d+\s*$", r"^\s*tape\s*\d+\s*-\s*tape\s*\d+\s*$", r"^\s*Page\s*\d+\s*$",
            r"^\s*DAY\s+\d+\s*$", r"^\s*Day\s+\d+\s*$", r"^\s*Day\s*$", r"^\s*Hour\s*$", r"^\s*Min\s*$", r"^\s*Sec\s*$",
            r"^\s*[A-Z][A-Z\s]+\(REV\s*\d+\)\s*$", r"^[»«<>\s]*END\s+OF\s+TAPE[»«<>\s]*$", r"^\s*MC\s+\d+\s*$", r"^\s*NOTE\s*$",
        ]
        if remove_numeric_only_lines:
            artifact_patterns.append(r"^\s*\d+\s*$")
        combined = "|".join(f"(?:{p})" for p in artifact_patterns)
        text = re.sub(combined, "\n", text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def parse_challenger(text: str, path: Path | None) -> list[dict[str, Any]]:
        meta_base = infer_doc_metadata(path)
        rows: list[dict[str, Any]] = []
        line_re = re.compile(r"^\[(?P<timestamp>\d+(?::\d+)+)\]\s+spk_(?P<speaker_id>\d+):\s*(?P<text>.*)$")
        for line in text.splitlines()[skiprows:]:
            m = line_re.match(line.strip())
            if not m:
                continue
            utterance_text = _clean_text(m.group("text"))
            if not utterance_text:
                continue
            timestamp = m.group("timestamp")
            speaker_id = int(m.group("speaker_id"))
            speaker = f"spk_{speaker_id}"
            utterance_index = len(rows)
            row_id = f"{meta_base['doc_id']}_utt_{str(utterance_index).zfill(6)}"
            metadata = make_metadata(meta_base, "challenger_diarized", utterance_index, speaker=speaker, speaker_id=speaker_id, timestamp=timestamp, timestamp_sec=colon_timestamp_to_seconds(timestamp))
            rows.append(make_row(row_id, utterance_text, metadata))
        return rows

    def parse_apollo(text: str, path: Path | None) -> list[dict[str, Any]]:
        meta_base = infer_doc_metadata(path)
        text = "".join(text.splitlines(keepends=True)[skiprows:])
        text = cut_to_start_page(text)
        text = remove_common_artifact_lines(text, remove_numeric_only_lines=False)
        timestamp_re = re.compile(r"^\s*(-?\d{2,3}\s+\d{2}\s+\d{2}\s+\d{2})\s*$")
        two_digit_re = re.compile(r"^\s*\d{2}\s*$")
        rows: list[dict[str, Any]] = []
        current_timestamp = current_speaker = ""
        buffer: list[str] = []
        pending_digits: list[str] = []
        pending_broken_timestamp = ""
        def flush() -> None:
            nonlocal buffer
            if not current_speaker or not buffer:
                buffer = []
                return
            utterance_text = _clean_text("\n".join(buffer))
            if not utterance_text:
                buffer = []
                return
            utterance_index = len(rows)
            row_id = f"{meta_base['doc_id']}_utt_{str(utterance_index).zfill(6)}"
            metadata = make_metadata(meta_base, "apollo_block", utterance_index, speaker=current_speaker, timestamp=current_timestamp, timestamp_sec=apollo_timestamp_to_seconds(current_timestamp))
            rows.append(make_row(row_id, utterance_text, metadata))
            buffer = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if two_digit_re.fullmatch(line):
                pending_digits.append(line)
                if len(pending_digits) == 4:
                    pending_broken_timestamp = " ".join(pending_digits)
                    pending_digits = []
                continue
            speaker_match = speaker_line_re.fullmatch(line)
            if speaker_match:
                speaker = normalize_speaker(speaker_match.group(1))
                if pending_broken_timestamp:
                    flush()
                    current_timestamp = pending_broken_timestamp
                    pending_broken_timestamp = ""
                flush()
                current_speaker = speaker
                continue
            pending_broken_timestamp = ""
            pending_digits = []
            timestamp_match = timestamp_re.fullmatch(line)
            if timestamp_match:
                flush()
                current_timestamp = timestamp_match.group(1)
                current_speaker = ""
                continue
            if current_speaker:
                buffer.append(line)
        flush()
        return rows

    def parse_pao(text: str, path: Path | None) -> list[dict[str, Any]]:
        meta_base = infer_doc_metadata(path)
        text = "".join(text.splitlines(keepends=True)[skiprows:])
        text = cut_to_start_page(text)
        text = remove_common_artifact_lines(text, remove_numeric_only_lines=True)
        rows: list[dict[str, Any]] = []
        current_speaker = current_segment_marker = ""
        buffer: list[str] = []
        segment_re = re.compile(r"^\s*(CDT\b.*GET\b.*|\d{1,2}/\d{1,2})\s*$", flags=re.IGNORECASE)
        pao_header_re = re.compile(r"^\s*(APOLLO\s+\d+\s+MISSION\s+COMMENTARY.*|APOLLO\s+\d+\s+STATUS\s+REPORT.*|APOLLO\s+\d+\s+SPACECRAFT\s+COMMENTARY.*|CDT\b.*GET\b.*|MC\s+\d+)\s*$", flags=re.IGNORECASE)
        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            utterance_text = _clean_text("\n".join(buffer))
            if not utterance_text:
                buffer = []
                return
            speaker = normalize_speaker(current_speaker or implicit_speaker or "PAO")
            utterance_index = len(rows)
            row_id = f"{meta_base['doc_id']}_utt_{str(utterance_index).zfill(6)}"
            metadata = make_metadata(meta_base, "pao_commentary", utterance_index, speaker=speaker, segment_marker=current_segment_marker)
            rows.append(make_row(row_id, utterance_text, metadata))
            buffer = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if pao_header_re.fullmatch(line):
                flush()
                current_segment_marker = line
                current_speaker = implicit_speaker
                continue
            if segment_re.fullmatch(line):
                flush()
                current_segment_marker = line
                current_speaker = implicit_speaker
                continue
            speaker_match = speaker_line_re.fullmatch(line)
            if speaker_match:
                flush()
                current_speaker = normalize_speaker(speaker_match.group(1))
                continue
            if not current_speaker:
                current_speaker = implicit_speaker
            buffer.append(line)
        flush()
        return rows

    def detect_parser(text: str, path: Path | None) -> str:
        if parser != "auto":
            return parser
        name = path.name.lower() if path is not None else source_file.lower()
        if re.search(r"(?m)^\[\d+(?::\d+)+\]\s+spk_\d+:", text):
            return "challenger"
        if "pao" in name or "commentary" in name:
            return "pao"
        if re.search(r"(?mi)MISSION\s+COMMENTARY|STATUS\s+REPORT|SPACECRAFT\s+COMMENTARY", text[:5000]):
            return "pao"
        return "apollo"

    all_rows: list[dict[str, Any]] = []
    if raw_text is not None:
        selected = detect_parser(raw_text, None)
        if selected == "challenger":
            all_rows.extend(parse_challenger(raw_text, None))
        elif selected == "apollo":
            all_rows.extend(parse_apollo(raw_text, None))
        elif selected == "pao":
            all_rows.extend(parse_pao(raw_text, None))
    else:
        if input_path is None:
            raise ValueError("input_path cannot be None when raw_text is not provided.")
        paths = sorted(input_path.glob(file_pattern)) if input_path.is_dir() else [input_path]
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            selected = detect_parser(text, path)
            if selected == "challenger":
                all_rows.extend(parse_challenger(text, path))
            elif selected == "apollo":
                all_rows.extend(parse_apollo(text, path))
            elif selected == "pao":
                all_rows.extend(parse_pao(text, path))

    columns = ["id", "document", "metadata"]
    if not all_rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(all_rows).reindex(columns=columns)
    df = df[df["document"].astype(str).str.strip().ne("")].reset_index(drop=True)
    return df


# =============================================================================
# Example build functions
# =============================================================================

def build_all_nasa_dataframes(base_dir: str | Path = "data_text") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function matching the folder layout discussed in the notebook.

    Returns:
        all_report_df, all_transcript_df, all_nasa_df
    """
    base_dir = Path(base_dir)

    mission_report_df = nasa_report_to_clean_df(
        file_path=base_dir / "apollo11" / "NASA_NTRS_Archive_19710015566_textract_full_text.txt",
        start_page=13,
        stop_heading="REFERENCES",
        mission="Apollo 11",
        collection="apollo11",
        report_type="mission_report",
        source_file="NASA_NTRS_Archive_19710015566_textract_full_text.txt",
    )

    flight_plan_df = nasa_report_to_clean_df(
        file_path=base_dir / "apollo11" / "Apollo_11_Flight_Plan_HSK_textract_full_text.txt",
        start_page=20,
        mission="Apollo 11",
        collection="apollo11",
        report_type="flight_plan",
        source_file="Apollo_11_Flight_Plan_HSK_textract_full_text.txt",
        drop_page_keywords=[
            "FIG", "FIGURE", "TABLE", "FORMS", "FROMS", "FORM", "LIST", "NOTES",
            "BURN CHART", "DATA CARD", "UPDATE PAD", "THIS PAGE INTENTIONALLY LEFT BLANK",
        ],
    )

    saturn_report_df = nasa_report_to_clean_df(
        file_path=base_dir / "apollo11" / "19900066485_textract_full_text.txt",
        start_page=29,
        mission="Apollo 11",
        collection="apollo11",
        report_type="saturn_v_flight_evaluation",
        source_file="19900066485_textract_full_text.txt",
    )

    all_report_df = pd.concat([mission_report_df, flight_plan_df, saturn_report_df], ignore_index=True)

    challenger_df = nasa_transcripts_to_clean_df(
        input_path=base_dir / "challenger",
        file_pattern="*Mission_Audio_transcript.txt",
        parser="challenger",
        mission="STS-51L",
        collection="challenger",
    )
    a11_pao_df = nasa_transcripts_to_clean_df(base_dir / "apollo11" / "a11transcript_pao_textract_full_text.txt", parser="pao", start_page=9, mission="Apollo 11", collection="apollo11")
    a11_tec_df = nasa_transcripts_to_clean_df(base_dir / "apollo11" / "a11transcript_tec_textract_full_text.txt", parser="apollo", start_page=11, mission="Apollo 11", collection="apollo11")
    a11_cm_df = nasa_transcripts_to_clean_df(base_dir / "apollo11" / "a11transscript_cm_textract_full_text.txt", parser="apollo", start_page=11, mission="Apollo 11", collection="apollo11")
    as13_pao_df = nasa_transcripts_to_clean_df(base_dir / "apollo13" / "AS13_PAO_textract_full_text.txt", parser="pao", start_page=1, mission="Apollo 13", collection="apollo13")
    as13_tec_df = nasa_transcripts_to_clean_df(base_dir / "apollo13" / "AS13_TEC_textract_full_text.txt", parser="apollo", start_page=8, mission="Apollo 13", collection="apollo13")
    as13_cm_df = nasa_transcripts_to_clean_df(base_dir / "apollo13" / "AS13_CM_textract_full_text.txt", parser="apollo", start_page=4, mission="Apollo 13", collection="apollo13")

    all_transcript_df = pd.concat(
        [challenger_df, a11_pao_df, a11_tec_df, a11_cm_df, as13_pao_df, as13_tec_df, as13_cm_df],
        ignore_index=True,
    )

    all_nasa_df = pd.concat([all_report_df, all_transcript_df], ignore_index=True)
    return all_report_df, all_transcript_df, all_nasa_df


if __name__ == "__main__":
    reports, transcripts, all_data = build_all_nasa_dataframes("data_text")
    print(build_debug_df(all_data).head(20))
