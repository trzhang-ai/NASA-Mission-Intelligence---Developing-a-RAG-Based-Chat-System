"""
NASA text parsing and cleaning helpers.

This module has two parsers:
1. nasa_report_to_clean_df: NASA report / flight-plan style OCR text.
2. nasa_transcripts_to_clean_df: NASA transcript / mission-commentary OCR text.

Both return a DataFrame with: id, document, metadata.
Use build_debug_df(df) for inspection. The output DataFrames can be chunked later.

Cleaning policy:
- Preserve the raw txt files; this module only creates derived retrieval records.
- Remove high-confidence structural OCR artifacts, not uncertain mission wording.
- Preserve page/line provenance so a retrieved statement can be traced to its source.
- Keep metadata values scalar so records remain compatible with ChromaDB.
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
    if not df["id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every id must be a non-empty string.")
    if not df["document"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Every document must be a non-empty string.")
    duplicate_count = df["id"].astype(str).duplicated().sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate ids.")

    common_metadata_keys = {
        "collection",
        "mission",
        "source",
        "source_type",
        "source_file",
        "source_path",
        "doc_id",
    }
    scalar_types = (str, int, float, bool)
    for row_number, metadata in enumerate(df["metadata"]):
        missing_metadata = common_metadata_keys - set(metadata)
        if missing_metadata:
            raise ValueError(
                f"Row {row_number} metadata is missing: {sorted(missing_metadata)}"
            )
        nonscalar = {
            key: type(value).__name__
            for key, value in metadata.items()
            if not isinstance(value, scalar_types)
        }
        if nonscalar:
            raise ValueError(
                f"Row {row_number} metadata has non-scalar values: {nonscalar}"
            )
        if metadata["source_type"] == "report":
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            if (
                not isinstance(page_start, int)
                or not isinstance(page_end, int)
                or page_start < 1
                or page_end < page_start
            ):
                raise ValueError(f"Row {row_number} has invalid report page bounds.")
        if metadata["source_type"] == "transcript":
            line_start = metadata.get("source_line_start")
            line_end = metadata.get("source_line_end")
            if (
                not isinstance(line_start, int)
                or not isinstance(line_end, int)
                or line_start < 1
                or line_end < line_start
            ):
                raise ValueError(f"Row {row_number} has invalid transcript line bounds.")
    duplicate_sources = df["metadata"].map(lambda value: value["source"]).duplicated().sum()
    if duplicate_sources:
        raise ValueError(
            f"Found {duplicate_sources} duplicate cleaner source-unit identifiers."
        )


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

    The legacy ``drop_pages_with_*`` argument names are retained for API
    compatibility. When enabled, mixed pages are filtered by narrative region;
    a single figure/table caption no longer deletes the entire page.
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
        raw_text = path.read_text(encoding="utf-8")
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
            r"MSC\s+FORM\s+\d+[A-Z]?",
        ]
        for pattern in common_header_patterns:
            page = re.sub(rf"(?mi)^\s*{pattern}\s*$", "", page)
        page = re.sub(r"(?m)^\s*#{2,}\s*$", "", page)
        return page.strip()

    def is_sentence_like_prose_line(line: str) -> bool:
        stripped = line.strip()
        if len(stripped) < 32:
            return False
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped)
        lowercase_count = sum(ch.islower() for ch in stripped)
        alpha_count = sum(ch.isalpha() for ch in stripped)
        nonspace_count = sum(not ch.isspace() for ch in stripped)
        return (
            len(words) >= 6
            and lowercase_count >= 10
            and nonspace_count > 0
            and alpha_count / nonspace_count >= 0.50
        )

    def is_visual_caption(line: str) -> bool:
        stripped = line.strip()
        if figure_caption_re.fullmatch(stripped):
            figure_match = re.match(
                r"(?i)^\s*(?:FIG|FIGURE)\.?\s+"
                r"[A-Z0-9]+(?:[-.][A-Z0-9]+)*\b(?P<rest>.*)$",
                stripped,
            )
            remainder = (
                figure_match.group("rest") if figure_match else ""
            ).strip()
            if re.match(r"^,\s*[a-z]", remainder):
                return False
            return True
        if not table_caption_re.fullmatch(stripped):
            return False
        table_match = re.match(
            r"(?i)^\s*TABLE\s+[A-Z0-9]+(?:[-.][A-Z0-9]+)*\b(?P<rest>.*)$",
            stripped,
        )
        remainder = (table_match.group("rest") if table_match else "").strip()
        # "Table 2-2 have been taken ..." is prose, while "TABLE 2-2. ..."
        # and "TABLE 2-2 PERFORMANCE DATA" are captions.
        if (
            remainder
            and remainder[0].islower()
            and is_sentence_like_prose_line(stripped)
        ):
            return False
        return True

    def is_high_confidence_visual_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.search(
            r"(?i)\bGROUND\s+TIME\s+MINUS\s+VEHICLE\s+TIME\b",
            stripped,
        ):
            return True
        if is_sentence_like_prose_line(stripped):
            return False
        if is_visual_caption(stripped):
            return True
        alpha_count = sum(ch.isalpha() for ch in stripped)
        digit_count = sum(ch.isdigit() for ch in stripped)
        punct_count = sum(
            not ch.isalnum() and not ch.isspace() for ch in stripped
        )
        alpha_words = re.findall(r"[A-Za-z]{2,}", stripped)
        if digit_count + punct_count > max(alpha_count, 3):
            return True
        if len(re.findall(r"\d+(?:\.\d+)?", stripped)) >= 3 and len(alpha_words) <= 4:
            return True
        if len(alpha_words) <= 2 and digit_count >= 2:
            return True
        if re.fullmatch(r"(?:\d{1,3}[A-Z]|[A-Z]{1,3}\d{1,3})", stripped):
            return True
        return False

    def visual_line_ratio(page: str) -> float:
        lines = [line for line in page.splitlines() if line.strip()]
        if not lines:
            return 1.0
        return sum(is_high_confidence_visual_line(line) for line in lines) / len(lines)

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
            if len(line) <= 28 or digit_count > alpha_count:
                short_or_numeric_lines += 1
            if digit_count + punct_count > alpha_count:
                numericish_lines += 1
            if is_sentence_like_prose_line(line):
                prose_lines += 1
        return prose_lines < 2 and (
            short_or_numeric_lines / len(lines) > 0.72 or numericish_lines / len(lines) > 0.55
        )

    def extract_narrative_regions(page: str) -> list[str]:
        """
        Reconstruct prose on mixed visual/text pages and omit chart/table OCR.

        OCR sometimes alternates a prose column with diagram labels line by line.
        Visual lines are therefore skipped without automatically splitting the prose.
        """
        lines = page.splitlines()
        narrative_terms = {
            "am", "is", "are", "was", "were", "be", "been", "being",
            "has", "have", "had", "do", "does", "did",
            "will", "would", "can", "could", "may", "might", "must", "should",
            "which", "that", "when", "while", "because", "therefore", "then",
        }
        function_words = {
            "a", "an", "the", "to", "of", "for", "by", "from",
            "that", "which", "when", "while", "because", "with", "as",
            "this", "these", "those",
        }

        def is_numbered_heading_candidate(line: str) -> bool:
            stripped = line.strip()
            if re.match(r"(?i)^SECTION\s+(?:[IVXLCDM]+|\d+)", stripped):
                return True
            match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", stripped)
            if not match:
                return False
            title = match.group(2).strip()
            if not title or not title[0].isupper():
                return False
            letters = re.findall(r"[A-Za-z]", title)
            if not letters:
                return False
            title_noise = sum(
                ch.isdigit() or (not ch.isalnum() and not ch.isspace())
                for ch in title
            )
            return (
                sum(ch.isupper() for ch in letters) / len(letters) >= 0.60
                and title_noise <= len(letters)
            )

        def pending_section_number(value: str) -> str:
            match = re.match(r"(?i)^SECTION\s+(\d+)\b", value)
            return match.group(1) if match else ""

        def is_short_uppercase_title_candidate(line: str) -> bool:
            stripped = line.strip()
            words = re.findall(r"[A-Za-z][A-Za-z/-]*", stripped)
            letters = re.findall(r"[A-Za-z]", stripped)
            return bool(
                1 <= len(words) <= 8
                and letters
                and not re.search(r"\d", stripped)
                and sum(ch.isupper() for ch in letters) / len(letters) >= 0.85
            )

        def is_narrative_candidate(line: str) -> bool:
            stripped = line.strip()
            if (
                not stripped
                or is_visual_caption(stripped)
                or is_high_confidence_visual_line(stripped)
            ):
                return False
            words = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped)
            if len(words) < 2:
                return False
            alpha_count = sum(ch.isalpha() for ch in stripped)
            nonspace_count = sum(not ch.isspace() for ch in stripped)
            if (
                nonspace_count == 0
                or alpha_count / nonspace_count < 0.55
                or sum(ch.islower() for ch in stripped) < 7
            ):
                return False
            normalized_words = {
                word.casefold().strip("-'") for word in words
            }
            normalized_word_sequence = [
                word.casefold().strip("-'") for word in words
            ]
            has_narrative_term = bool(normalized_words & narrative_terms)
            has_inflected_word = any(
                len(word) >= 5
                and word.casefold().strip("-'").endswith(("ed", "ing"))
                for word in words
            )
            has_sentence_boundary = bool(re.search(r"[.!?;:](?:\s|$)", stripped))
            has_ocr_continuation = stripped.endswith("-")
            function_word_count = sum(
                word in function_words for word in normalized_word_sequence
            )
            has_such_as = bool(re.search(r"(?i)\bsuch\s+as\b", stripped))
            is_long_lowercase_continuation = (
                len(words) >= 6 and stripped[0].islower()
            )
            is_hyphenated_lowercase_continuation = (
                len(words) >= 4
                and stripped[0].islower()
                and any("-" in word.strip("-") for word in words)
            )
            is_short_participial_fragment = (
                len(words) <= 3
                and normalized_word_sequence[0].endswith("ing")
                and not has_narrative_term
                and function_word_count == 0
            )
            if is_short_participial_fragment:
                return False
            return (
                has_narrative_term
                or (has_inflected_word and len(words) >= 4)
                or has_sentence_boundary
                or (has_ocr_continuation and len(words) >= 3)
                or function_word_count >= 4
                or has_such_as
                or is_long_lowercase_continuation
                or is_hyphenated_lowercase_continuation
            )

        regions: list[str] = []
        current_lines: list[str] = []
        pending_heading = ""
        skipped_since_prose = 0

        def flush_region() -> None:
            nonlocal current_lines, pending_heading, skipped_since_prose
            if current_lines:
                prose_lines = [
                    line
                    for line in current_lines
                    if not is_numbered_heading_candidate(line)
                ]
                alpha_count = sum(
                    ch.isalpha() for line in prose_lines for ch in line
                )
                if alpha_count >= 60:
                    regions.append("\n".join(current_lines).strip())
            current_lines = []
            pending_heading = ""
            skipped_since_prose = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_region()
                continue
            if (
                pending_section_number(pending_heading)
                and "\n" not in pending_heading
                and is_short_uppercase_title_candidate(stripped)
            ):
                pending_heading = f"{pending_heading}\n{stripped}"
                continue
            if is_visual_caption(stripped):
                flush_region()
                continue
            if is_numbered_heading_candidate(stripped):
                section_number = pending_section_number(pending_heading)
                numbered_match = re.match(r"^(\d+(?:\.\d+)*)", stripped)
                if (
                    section_number
                    and numbered_match
                    and numbered_match.group(1).split(".")[0] == section_number
                ):
                    pending_heading = f"{pending_heading}\n{stripped}"
                    continue
                flush_region()
                pending_heading = stripped
                continue
            if is_narrative_candidate(stripped):
                if not current_lines and pending_heading:
                    current_lines.append(pending_heading)
                current_lines.append(stripped)
                skipped_since_prose = 0
                continue
            if current_lines:
                skipped_since_prose += 1
                if skipped_since_prose > 8:
                    flush_region()
        flush_region()
        return regions

    kept_pages: list[tuple[int, str, int]] = []
    merge_group = 0
    previous_page_num: int | None = None
    previous_was_complete_page = False
    for page_num, page in pages:
        if page_num < start_page:
            continue
        stop_after_page = False
        if stop_heading:
            stop_match = re.search(
                rf"(?mi)^\s*{re.escape(stop_heading)}\s*$",
                page,
            )
            if stop_match:
                page = page[:stop_match.start()]
                stop_after_page = True
        page = strip_page_noise(page)
        if not page:
            previous_was_complete_page = False
            previous_page_num = page_num
            if stop_after_page:
                break
            continue

        lines = [line for line in page.splitlines() if line.strip()]
        has_figure_caption = drop_pages_with_figures and any(
            figure_caption_re.fullmatch(line.strip())
            for line in lines
        )
        has_table_caption = drop_pages_with_tables and any(
            table_caption_re.fullmatch(line.strip())
            and is_visual_caption(line)
            for line in lines
        )
        keyword_hint = page_has_drop_keyword(page, drop_page_keywords)
        visually_dominated = looks_like_visual_or_table_body(page)
        needs_region_extraction = (
            has_figure_caption
            or has_table_caption
            or visually_dominated
            or (keyword_hint and visual_line_ratio(page) >= 0.35)
        )

        if needs_region_extraction:
            regions = extract_narrative_regions(page)
            for region in regions:
                merge_group += 1
                kept_pages.append((page_num, region, merge_group))
            previous_was_complete_page = False
        else:
            if not (
                previous_was_complete_page
                and previous_page_num is not None
                and page_num == previous_page_num + 1
            ):
                merge_group += 1
            kept_pages.append((page_num, page, merge_group))
            previous_was_complete_page = not stop_after_page
        previous_page_num = page_num
        if stop_after_page:
            break

    line_items: list[tuple[int, str, int]] = []
    for page_num, page, group in kept_pages:
        for line in page.splitlines():
            line_items.append((page_num, line, group))
        line_items.append((page_num, "", group))

    def roman_to_int(s: str) -> int | None:
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
        total = previous = 0
        for char in reversed(s.upper()):
            value = values.get(char, 0)
            total = total - value if value < previous else total + value
            previous = max(previous, value)
        return total or None

    visual_axis_label_re = re.compile(
        r"(?i)^(?:"
        r"(?:rms\s+)?amplitude\s*[.,:/-]?\s*arbitrary\s+units?"
        r"|"
        r"\d+(?:\.\d+)?\s*(?:fps|hz|sec|min|hr|deg|m|km|ft|psi|dbm?)"
        r"|"
        r"(?:frequency|time|amplitude|altitude|velocity|temperature|"
        r"pressure|distance|range|latitude|longitude|pitch|yaw|roll)"
        r"\s*[.,:/-]?\s*"
        r"(?:hz|seconds?|sec|min(?:utes?)?|hours?|hr|degrees?|deg|"
        r"meters?|m|kilometers?|km|feet|ft|volts?|amps?|psi|"
        r"dbm?|percent|units?)"
        r")$"
    )

    def is_visual_axis_label(line: str) -> bool:
        return bool(visual_axis_label_re.fullmatch(line.strip()))

    def parse_heading(line: str) -> dict[str, Any] | None:
        s = line.strip()
        if not s or len(s) > 140:
            return None
        if is_visual_axis_label(s):
            return None
        if re.search(r"(?i)\bTHIS PAGE INTENTIONALLY LEFT BLANK\b", s):
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
            if re.fullmatch(
                r"(?i)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"\.?\s+\d{2,4}\s+\d{3,4}\s+"
                r"(?:UTC|GMT|[ECMP][SD]T)",
                title,
            ):
                return None
            if not re.search(r"[A-Za-z]", title):
                return None
            if not re.match(r"^(?:[-–]\s*)?[A-Za-z]", title):
                return None
            title_alpha = sum(ch.isalpha() for ch in title)
            title_noise = sum(
                ch.isdigit() or (not ch.isalnum() and not ch.isspace())
                for ch in title
            )
            if title_noise > title_alpha:
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
        if re.search(r"\d", s) or re.search(r"[()=+]", s):
            return None
        letters = re.findall(r"[A-Za-z]", s)
        if not letters:
            return None
        if len(words) == 1:
            allowed_single_word_headings = {
                "INTRODUCTION",
                "SUMMARY",
                "CONCLUSIONS",
                "REFERENCES",
                "APPENDIX",
                "APPROVAL",
                "CONTENTS",
                "TRAJECTORY",
                "STRUCTURES",
                "SEPARATION",
            }
            if s.upper() not in allowed_single_word_headings:
                return None
        first_alpha = next((ch for ch in s if ch.isalpha()), "")
        if first_alpha and not first_alpha.isupper():
            return None
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        cap_words = sum(word[0].isupper() for word in words)
        is_all_caps_heading = upper_ratio >= 0.82 and len(words) <= 8
        is_short_title_heading = len(words) <= 5 and cap_words == len(words)
        if is_all_caps_heading or is_short_title_heading:
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

    def is_low_information_block(block: str) -> bool:
        alpha_count = sum(ch.isalpha() for ch in block)
        digit_count = sum(ch.isdigit() for ch in block)
        punct_count = sum(
            not ch.isalnum() and not ch.isspace() for ch in block
        )
        alpha_words = re.findall(r"[A-Za-z]{2,}", block)
        return not alpha_words or (
            len(alpha_words) < 5
            and digit_count + punct_count > max(alpha_count, 1)
        )

    def retain_narrative_from_noisy_block(block: str) -> str:
        """
        Salvage an explanatory note/sentence from a numeric block, or reject the block.

        This is intentionally density-based. It does not alter words or invent missing
        punctuation; it only selects a source substring already present in the OCR.
        """
        alpha_count = sum(ch.isalpha() for ch in block)
        noise_count = sum(
            ch.isdigit() or (not ch.isalnum() and not ch.isspace())
            for ch in block
        )
        if not alpha_count:
            return ""
        noise_ratio = noise_count / alpha_count

        marker_match = re.search(r"(?i)\bNOTE:\s*|(?<!\*)\*{1,2}(?=[A-Za-z])", block)
        if marker_match and noise_ratio >= 0.40:
            candidate = block[marker_match.start():].lstrip("* ").strip()
            candidate_alpha = sum(ch.isalpha() for ch in candidate)
            candidate_noise = sum(
                ch.isdigit() or (not ch.isalnum() and not ch.isspace())
                for ch in candidate
            )
            if (
                len(re.findall(r"[A-Za-z]{2,}", candidate)) >= 4
                and candidate_alpha
                and candidate_noise / candidate_alpha <= 0.55
            ):
                return candidate

        if noise_ratio <= 0.60:
            return block

        for word_match in re.finditer(r"\b[A-Za-z][A-Za-z'-]{2,}\b", block):
            candidate = block[word_match.start():].strip()
            candidate_alpha = sum(ch.isalpha() for ch in candidate)
            candidate_noise = sum(
                ch.isdigit() or (not ch.isalnum() and not ch.isspace())
                for ch in candidate
            )
            if (
                len(re.findall(r"[A-Za-z]{2,}", candidate)) >= 8
                and sum(ch.islower() for ch in candidate) >= 15
                and candidate_alpha
                and candidate_noise / candidate_alpha <= 0.45
            ):
                return candidate
        return ""

    blocks: list[tuple[str, int, int, int]] = []
    buffer: list[str] = []
    buffer_pages: list[int] = []
    buffer_group: int | None = None

    def flush_block_buffer() -> None:
        nonlocal buffer, buffer_pages, buffer_group
        if buffer:
            blocks.append(
                (
                    " ".join(buffer),
                    min(buffer_pages),
                    max(buffer_pages),
                    int(buffer_group),
                )
            )
        buffer = []
        buffer_pages = []
        buffer_group = None

    for page_num, raw_line, group in line_items:
        if buffer and buffer_group != group:
            flush_block_buffer()
        line = raw_line.strip()
        if not line:
            flush_block_buffer()
            continue
        if re.fullmatch(r"[A-Z]?\d+-\d+(?:/[A-Z]?\d+-\d+)?", line) or re.fullmatch(r"\d+", line):
            continue
        if is_visual_axis_label(line):
            continue
        if is_heading(line):
            flush_block_buffer()
            blocks.append((line, page_num, page_num, group))
        else:
            if buffer_group is None:
                buffer_group = group
            buffer.append(line)
            buffer_pages.append(page_num)
    flush_block_buffer()
    cleaned_blocks: list[tuple[str, int, int, int]] = []
    for block, page_start, page_end, group in blocks:
        cleaned = clean_block(block)
        if not cleaned:
            continue
        if re.search(r"(?i)\bTHIS PAGE INTENTIONALLY LEFT BLANK\b", cleaned):
            continue
        if not is_heading(cleaned):
            cleaned = retain_narrative_from_noisy_block(cleaned)
        if cleaned and (is_heading(cleaned) or not is_low_information_block(cleaned)):
            cleaned_blocks.append((cleaned, page_start, page_end, group))
    blocks = cleaned_blocks

    merged_blocks: list[tuple[str, int, int, int]] = []
    for block, page_start, page_end, group in blocks:
        if (
            merged_blocks
            and not is_heading(merged_blocks[-1][0])
            and not is_heading(block)
            and (
                merged_blocks[-1][3] == group
                or merged_blocks[-1][2] == page_start
            )
            and ends_mid_sentence(merged_blocks[-1][0])
            and starts_like_continuation(block)
        ):
            previous_block, previous_start, previous_end, previous_group = merged_blocks[-1]
            merged_blocks[-1] = (
                clean_block(previous_block + " " + block),
                min(previous_start, page_start),
                max(previous_end, page_end),
                previous_group,
            )
        else:
            merged_blocks.append((block, page_start, page_end, group))

    rows: list[dict[str, Any]] = []
    section_stack: list[dict[str, Any]] = []
    pending_section: dict[str, Any] | None = None
    for block, page_start, page_end, _group in merged_blocks:
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
                is_appendix_heading = bool(
                    re.fullmatch(
                        r"(?i)APPENDIX(?:\s+[A-Z0-9]+)?",
                        heading["title"],
                    )
                )
                appendix_rooted = bool(
                    section_stack
                    and re.fullmatch(
                        r"(?i)APPENDIX(?:\s+[A-Z0-9]+)?",
                        section_stack[0]["title"],
                    )
                )
                if is_appendix_heading:
                    section_stack = [heading]
                elif appendix_rooted:
                    section_stack = section_stack[:1]
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
        row_id = f"{doc_id}_report_{len(rows):05d}"
        metadata["source"] = row_id
        rows.append({"id": row_id, "document": document, "metadata": metadata})

    df = pd.DataFrame(rows, columns=["id", "document", "metadata"])
    validate_cleaned_df(df)
    return df


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
    """
    Parse NASA transcript txt files into a Chroma-ready DataFrame.

    Timestamp and speaker structure is normalized, but dialogue wording is not
    rewritten. Invalid OCR timestamps are retained with ``timestamp_valid=False``.
    If ``start_page`` is requested and its marker is absent, parsing raises a clear
    error instead of silently processing the wrong portion of the file.
    """
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
            if minutes < 0 or not 0 <= seconds < 60:
                return None
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
                return None
            return hours * 3600 + minutes * 60 + seconds
        return None

    def apollo_timestamp_info(ts: str | None) -> tuple[int | None, bool]:
        if not ts:
            return None, False
        match = re.fullmatch(
            r"(?P<sign>-?)(?P<day>\d{2,3})\s+"
            r"(?P<hour>\d{2})\s+(?P<minute>\d{2})\s+(?P<second>\d{2})",
            ts.strip(),
        )
        if not match:
            return None, False
        day = int(match.group("day"))
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        second = int(match.group("second"))
        if hour >= 24 or minute >= 60 or second >= 60:
            return None, False
        total_seconds = ((day * 24 + hour) * 60 + minute) * 60 + second
        if match.group("sign") == "-":
            total_seconds = -total_seconds
        return total_seconds, True

    page_marker_re = re.compile(r"^\s*---\s*PAGE\s*(\d+)\s*---\s*$", re.IGNORECASE)
    mission_voice_title_fragment_re = re.compile(
        r"(?i)(?:\d+\s+)?"
        r"(?:\(GOSS\s+NET\s+\d+\)\s*)?"
        r"(?:A\s*POLLO|APOLLO)\s+(?:\d+|B)(?:\s*-\s*|\s+)"
        r"(?:TECHNICAL\s+)?"
        r"(?:(?:AIR|ATR)-TO-(?:GROUND|GROUID)|ONBOARD)\s+VOICE\s+"
        r"(?:TRANSCRIPTION|TRANCCRIPTION)"
        r"(?:\s*\(GOSS\s+NET\s+\d+\))?"
    )
    voice_title_fragment_re = re.compile(
        r"(?i)(?:\d+\s+)?"
        r"(?:\(GOSS\s+NET\s+\d+\)\s*)?"
        r"(?:(?:A\s*POLLO|APOLLO)\s+(?:\d+|B)(?:\s*-\s*|\s+))?"
        r"(?:TECHNICAL\s+)?"
        r"(?:(?:AIR|ATR)-TO-(?:GROUND|GROUID)|ONBOARD)\s+VOICE\s+"
        r"(?:TRANSCRIPTION|TRANCCRIPTION)"
        r"(?:\s*\(GOSS\s+NET\s+\d+\))?"
    )
    goss_net_fragment_re = re.compile(
        r"(?i)(?:\b\d+\s+)?\(\s*GOSS\s+NET\s+\d+\s*\)"
    )
    end_of_tape_fragment_re = re.compile(
        r"(?i)[»«<>]*\s*END\s+OF\s+TAPE\s*[»«<>]*"
    )
    ocr_voice_title_signature_re = re.compile(
        r"(?i)\b(?:\d{2}|B)\b.*"
        r"(?:AIR|ATR)-TO-(?:GROUND|GROUID)\s+VOICE\b"
    )
    tape_header_re = re.compile(
        r"(?i)^\s*Tapes?\s+[A-Z0-9]+(?:-[A-Z0-9]+)?/[A-Z0-9]+\s*$"
    )

    def prepare_source_lines(
        text: str,
        *,
        apply_start_page: bool,
    ) -> list[tuple[int, int | None, str]]:
        """
        Preserve original 1-based line numbers and page numbers while selecting input.

        Page-marker lines are structural metadata, so they are not returned as dialogue.
        """
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        indexed_lines = list(enumerate(normalized.splitlines(), start=1))[skiprows:]
        selected: list[tuple[int, int | None, str]] = []
        current_page: int | None = None
        found_start = not apply_start_page or start_page is None
        for line_number, line in indexed_lines:
            page_match = page_marker_re.fullmatch(line)
            if page_match:
                current_page = int(page_match.group(1))
                if apply_start_page and start_page == current_page:
                    found_start = True
                continue
            if found_start:
                selected.append((line_number, current_page, line))
        if apply_start_page and start_page is not None and not found_start:
            raise ValueError(f"Requested start page {start_page} was not found in the transcript.")

        without_split_headers: list[tuple[int, int | None, str]] = []

        def next_nonblank_index(after_index: int) -> int | None:
            candidate = after_index + 1
            while (
                candidate < len(selected)
                and not selected[candidate][2].strip()
            ):
                candidate += 1
            return candidate if candidate < len(selected) else None

        def is_ocr_voice_title_before_tape(candidate_index: int) -> bool:
            if not ocr_voice_title_signature_re.search(
                selected[candidate_index][2]
            ):
                return False
            following_index = next_nonblank_index(candidate_index)
            return bool(
                following_index is not None
                and tape_header_re.fullmatch(selected[following_index][2])
            )

        index = 0
        while index < len(selected):
            if is_ocr_voice_title_before_tape(index):
                without_split_headers.append(
                    (selected[index][0], selected[index][1], "")
                )
                index += 1
                continue

            if re.fullmatch(r"(?i)\s*AP\s*", selected[index][2]):
                title_index = next_nonblank_index(index)
                if (
                    title_index is not None
                    and is_ocr_voice_title_before_tape(title_index)
                ):
                    without_split_headers.append(
                        (selected[index][0], selected[index][1], "")
                    )
                    index += 1
                    continue

            four_lines = selected[index:index + 4]
            if (
                len(four_lines) == 4
                and all(
                    re.fullmatch(r"\s*\d{2}\s*", item[2])
                    for item in four_lines
                )
            ):
                next_index = index + 4
                while (
                    next_index < len(selected)
                    and not selected[next_index][2].strip()
                ):
                    next_index += 1
                if (
                    next_index < len(selected)
                    and speaker_line_re.fullmatch(selected[next_index][2])
                ):
                    without_split_headers.append(
                        (
                            four_lines[0][0],
                            four_lines[0][1],
                            " ".join(item[2].strip() for item in four_lines),
                        )
                    )
                    index += 4
                    continue

            goss_start = re.search(
                r"(?i)\(\s*GOSS\s*$",
                selected[index][2],
            )
            if goss_start and index + 1 < len(selected):
                goss_sequence_length = 0
                if (
                    index + 2 < len(selected)
                    and re.fullmatch(r"(?i)\s*NET\s*", selected[index + 1][2])
                    and re.fullmatch(r"\s*\d+\s*\)\s*", selected[index + 2][2])
                ):
                    goss_sequence_length = 3
                elif re.fullmatch(
                    r"(?i)\s*NET\s+\d+\s*\)\s*",
                    selected[index + 1][2],
                ):
                    goss_sequence_length = 2
                if goss_sequence_length:
                    line_number, page_number, line = selected[index]
                    line = line[:goss_start.start()]
                    line = mission_voice_title_fragment_re.sub(" ", line)
                    line = voice_title_fragment_re.sub(" ", line)
                    without_split_headers.append(
                        (line_number, page_number, line)
                    )
                    index += goss_sequence_length
                    continue

            five_lines = selected[index:index + 5]
            if (
                len(five_lines) == 5
                and re.fullmatch(r"(?i)\s*Tape\s*", five_lines[0][2])
                and re.fullmatch(r"\s*\d+/\d+\s*", five_lines[1][2])
                and re.fullmatch(r"(?i)\s*(?:-|I|\|)?\s*", five_lines[2][2])
                and re.fullmatch(r"(?i)\s*(?:Page|Fage)\s*", five_lines[3][2])
                and re.fullmatch(r"\s*\d+\s*", five_lines[4][2])
            ):
                without_split_headers.append(
                    (selected[index][0], selected[index][1], "")
                )
                index += 5
                continue

            sequence_start = index
            if re.fullmatch(r"\s*[»«<>]+\s*", selected[index][2]):
                sequence_start += 1
            if sequence_start + 2 < len(selected):
                end_word = re.sub(r"[»«<>]", "", selected[sequence_start][2]).strip()
                of_word = re.sub(r"[»«<>]", "", selected[sequence_start + 1][2]).strip()
                tape_word = re.sub(r"[»«<>]", "", selected[sequence_start + 2][2]).strip()
                if (
                    end_word.casefold() == "end"
                    and of_word.casefold() == "of"
                    and tape_word.casefold() == "tape"
                ):
                    sequence_end = sequence_start + 3
                    if (
                        sequence_end < len(selected)
                        and re.fullmatch(r"\s*[»«<>]+\s*", selected[sequence_end][2])
                    ):
                        sequence_end += 1
                    without_split_headers.append(
                        (selected[index][0], selected[index][1], "")
                    )
                    index = sequence_end
                    continue

            line_number, page_number, line = selected[index]
            if line.strip().isdigit():
                next_nonblank = index + 1
                while (
                    next_nonblank < len(selected)
                    and not selected[next_nonblank][2].strip()
                ):
                    next_nonblank += 1
                if (
                    next_nonblank < len(selected)
                    and (
                        mission_voice_title_fragment_re.search(
                            selected[next_nonblank][2]
                        )
                        or voice_title_fragment_re.search(
                            selected[next_nonblank][2]
                        )
                    )
                ):
                    without_split_headers.append((line_number, page_number, ""))
                    index += 1
                    continue
            line = end_of_tape_fragment_re.sub(" ", line)
            line = mission_voice_title_fragment_re.sub(" ", line)
            line = voice_title_fragment_re.sub(" ", line)
            line = goss_net_fragment_re.sub(" ", line)
            without_split_headers.append((line_number, page_number, line))
            index += 1
        return without_split_headers

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

    def make_metadata(
        meta_base: dict[str, str],
        transcript_format: str,
        utterance_index: int,
        speaker: str = "",
        timestamp: str = "",
        timestamp_sec: int | None = None,
        timestamp_valid: bool | None = None,
        speaker_id: int | None = None,
        segment_marker: str = "",
        source_line_start: int | None = None,
        source_line_end: int | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        speaker_label_status: str = "",
        calendar_date: str = "",
    ) -> dict[str, Any]:
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
        if timestamp_valid is not None:
            metadata["timestamp_valid"] = bool(timestamp_valid)
        if segment_marker:
            metadata["segment_marker"] = segment_marker
        if source_line_start is not None:
            metadata["source_line_start"] = int(source_line_start)
        if source_line_end is not None:
            metadata["source_line_end"] = int(source_line_end)
        if page_start is not None:
            metadata["page_start"] = int(page_start)
        if page_end is not None:
            metadata["page_end"] = int(page_end)
        if speaker_label_status:
            metadata["speaker_label_status"] = speaker_label_status
        if calendar_date:
            metadata["calendar_date"] = calendar_date
        return metadata

    def make_row(row_id: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        prepared_metadata = metadata.copy()
        prepared_metadata["source"] = row_id
        return {
            "id": row_id,
            "document": _clean_text(text),
            "metadata": prepared_metadata,
        }

    def normalize_speaker(s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip().upper())
        return {"P AO": "PAO", "P A O": "PAO", "CAP COM": "CAPCOM", "CAPCOMM": "CAPCOM"}.get(s, s)

    speaker_names = [
        "PAO", "P AO", "P A O", "MCC", "MSC", "KSC", "LCC", "CAPCOM", "CAP COM", "COMM TECH",
        "CDR", "CMP", "LMP", "CDR/LMP", "CDR/CMP", "CDR-CMP", "CMP/LMP", "LMP/CDR", "CMP/CDR",
        "CDR-EVA", "LMP-EVA", "CDR-LM", "LMP-LM", "SC", "MS", "CC", "CT", "MSFN", "F",
        "HORNET", "R", "R-1", "R-2", "R-3", "AB", "IWO", "P-1", "P-2", "CMF", "LMF", "IMP",
        "CDR (EAGLE)", "LMP (EAGLE)", "CMP (COLUMBIA)", "CDR (TRANQ)", "LMP (TRANQ)",
        "CDR (EVA)", "LMP (EVA)", "SWIM 1", "S-1", "S-2", "CDF", "LMP/CMP",
        "PRESIDENT NIXON",
    ]
    ocr_speaker_labels = {
        "TMP",
        "CMH",
        "C :",
        "MP",
        "CMI'",
        "LMI",
        "LMU",
        "LMI'",
    }
    speaker_names.extend(ocr_speaker_labels)
    if speakers:
        speaker_names.extend(speakers)
    speaker_names = sorted(set(speaker_names), key=len, reverse=True)
    speaker_token_pattern = "|".join(re.escape(name) for name in speaker_names)
    speaker_line_re = re.compile(
        rf"^\s*({speaker_token_pattern})\s*$",
        flags=re.IGNORECASE,
    )

    artifact_patterns = [
        r"\s*#{2,}\s*",
        r"\s*(CONFIDENTIAL|CONFIDENTIA|CONFIDENTI|UNCLASSIFIED)\s*",
        r"\s*NATIONAL AERONAUTICS AND SPACE ADMINISTRATION\s*",
        r"\s*MANNED SPACECRAFT CENTER\s*",
        r"\s*HOUSTON,?\s*TEXAS\s*",
        r"\s*(?:\d+\s+)?(?:\(GOSS\s+NET\s+\d+\)\s*)?APOLLO\s+\d+\s*-?\s*AIR-TO-GROUND\s+VOICE\s+TRANSCRIPTION(?:\s*\(GOSS\s+NET\s+\d+\))?\s*",
        r"\s*(?:\d+\s+)?(?:\(GOSS\s+NET\s+\d+\)\s*)?APOLLO\s+\d+\s*-?\s*ONBOARD\s+VOICE\s+TRANSCRIPTION(?:\s*\(GOSS\s+NET\s+\d+\))?\s*",
        r"\s*A\s*POLLO\s+\d+\s*-?\s*MISSION\s+COMMENTARY.*",
        r"\s*APOLLO\s+\d+\s*-?\s*MISSION\s+COMMENTARY\s*",
        r"\s*APOLLO\s+\d+\s+MISSION\s+COMMENTARY.*",
        r"\s*APOLLO\s+\d+\s+STATUS\s+REPORT.*",
        r"\s*APOLLO\s+\d+\s+SPACECRAFT\s+COMMENTARY.*",
        r"\s*Tapes?\s*[A-Z0-9]+(?:-[A-Z0-9]+)?/[A-Z0-9]+\s*(?:-?\s*(?:Page|Fage|PAPER|Prive|Pragre|Payte(?:\s+only)?|Progre(?:ss)?|Prote)\s*\d*)?\s*",
        r"\s*tape\s*\d+\s*-\s*tape\s*\d+\s*",
        r"\s*(?:Page|Fage|PAPER|Prive|Pragre|Payte(?:\s+only)?)\s*\d*\s*",
        r"\s*DAY\s+\d+\s*",
        r"\s*Day\s*",
        r"\s*Hour\s*",
        r"\s*Min\s*",
        r"\s*Sec\s*",
        r"\s*[A-Z][A-Z\s]+\(REV\s*\d+\)\s*",
        r"[»«<>\s]*END\s+OF\s+TAPE[»«<>\s]*",
        r"\s*MC\s+\d+\s*",
        r"\s*NOTE\s*",
    ]
    artifact_line_res = [
        re.compile(pattern, re.IGNORECASE) for pattern in artifact_patterns
    ]

    def is_common_artifact_line(line: str, remove_numeric_only_lines: bool) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if remove_numeric_only_lines and stripped.isdigit():
            return True
        return any(pattern.fullmatch(stripped) for pattern in artifact_line_res)

    def parse_challenger(text: str, path: Path | None) -> list[dict[str, Any]]:
        meta_base = infer_doc_metadata(path)
        rows: list[dict[str, Any]] = []
        line_re = re.compile(r"^\[(?P<timestamp>\d+(?::\d+)+)\]\s+spk_(?P<speaker_id>\d+):\s*(?P<text>.*)$")
        current: dict[str, Any] | None = None

        def flush() -> None:
            nonlocal current
            if current is None:
                return
            utterance_text = _clean_text("\n".join(current["text_lines"]))
            if utterance_text:
                utterance_index = len(rows)
                row_id = f"{meta_base['doc_id']}_utt_{str(utterance_index).zfill(6)}"
                timestamp_sec = colon_timestamp_to_seconds(current["timestamp"])
                metadata = make_metadata(
                    meta_base,
                    "challenger_diarized",
                    utterance_index,
                    speaker=current["speaker"],
                    speaker_id=current["speaker_id"],
                    timestamp=current["timestamp"],
                    timestamp_sec=timestamp_sec,
                    timestamp_valid=timestamp_sec is not None,
                    source_line_start=current["line_start"],
                    source_line_end=current["line_end"],
                    page_start=current["page_start"],
                    page_end=current["page_end"],
                )
                rows.append(make_row(row_id, utterance_text, metadata))
            current = None

        for line_number, page_number, raw_line in prepare_source_lines(
            text,
            apply_start_page=False,
        ):
            line = raw_line.strip()
            if not line or is_common_artifact_line(line, remove_numeric_only_lines=False):
                continue
            match = line_re.match(line)
            if match:
                flush()
                speaker_id = int(match.group("speaker_id"))
                current = {
                    "timestamp": match.group("timestamp"),
                    "speaker_id": speaker_id,
                    "speaker": f"spk_{speaker_id}",
                    "text_lines": [match.group("text")] if match.group("text").strip() else [],
                    "line_start": line_number,
                    "line_end": line_number,
                    "page_start": page_number,
                    "page_end": page_number,
                }
                continue
            if current is not None:
                current["text_lines"].append(line)
                current["line_end"] = line_number
                if page_number is not None:
                    if current["page_start"] is None:
                        current["page_start"] = page_number
                    current["page_end"] = page_number
        flush()
        return rows

    def parse_apollo(text: str, path: Path | None) -> list[dict[str, Any]]:
        meta_base = infer_doc_metadata(path)
        timestamp_token = r"-?\d{2,3}\s+\d{2}\s+\d{2}\s+\d{2}"
        timestamp_re = re.compile(rf"^\s*(?P<timestamp>{timestamp_token})\s*$")
        ocr_day_token = (
            r"(?=[A-Z0-9%?]*[A-Z%?])[A-Z0-9%?]{2,3}"
        )
        ocr_timestamp_re = re.compile(
            rf"^\s*(?P<timestamp>{ocr_day_token}\s+\d{{2}}\s+"
            rf"\d{{2}}\s+\d{{2}})\s*$",
            re.IGNORECASE,
        )
        spaced_fused_re = re.compile(
            rf"^\s*(?P<timestamp>{timestamp_token})\s+"
            rf"(?P<speaker>{speaker_token_pattern})"
            r"(?:\s+(?P<text>.+))?\s*$",
            re.IGNORECASE,
        )
        no_space_fused_re = re.compile(
            rf"^\s*(?P<timestamp>{timestamp_token})"
            r"(?P<speaker>CDR|CMP|LMP|CC)"
            r"\s*(?P<text>(?:[A-Z][a-z]|I(?:\s|')|\.\.\.).*)\s*$"
        )
        no_space_speaker_only_re = re.compile(
            rf"^\s*(?P<timestamp>{timestamp_token})"
            r"(?P<speaker>CDR|CMP|LMP|CC)\s*$"
        )
        joined_speaker_re = re.compile(
            r"^(?P<speaker>CDR|CMP|LMP|CC)"
            r"(?P<text>(?:[A-Z][a-z]|I(?:\s|')|\.\.\.|"
            r"[A-Z]{2,}[A-Z .'-]*$).*)$"
        )
        awaiting_speaker_names = [
            name
            for name in speaker_names
            if len(re.sub(r"[^A-Za-z]", "", name)) >= 2
        ]
        awaiting_speaker_pattern = "|".join(
            re.escape(name) for name in awaiting_speaker_names
        )
        awaiting_speaker_with_text_re = re.compile(
            rf"^(?P<speaker>{awaiting_speaker_pattern})\s+"
            r"(?P<text>\S.*)$"
        )
        president_with_text_re = re.compile(
            r"^(?P<speaker>PRESIDENT NIXON)\s+(?P<text>.+)$"
        )
        two_digit_re = re.compile(r"^\s*\d{2}\s*$")
        rows: list[dict[str, Any]] = []
        current_timestamp = ""
        current_timestamp_sec: int | None = None
        current_timestamp_valid: bool | None = None
        current_speaker = ""
        current_speaker_status = ""
        buffer: list[tuple[int, int | None, str]] = []
        record_start_line: int | None = None
        record_start_page: int | None = None
        awaiting_speaker = False
        pending_digits: list[tuple[int, int | None, str]] = []

        def speaker_status(speaker: str) -> str:
            return (
                "ocr_alias"
                if normalize_speaker(speaker) in ocr_speaker_labels
                else "recognized"
            )

        def flush() -> None:
            nonlocal buffer, record_start_line, record_start_page
            if not current_speaker or not buffer:
                buffer = []
                record_start_line = None
                record_start_page = None
                return
            utterance_text = _clean_text("\n".join(item[2] for item in buffer))
            if not utterance_text:
                buffer = []
                record_start_line = None
                record_start_page = None
                return
            utterance_index = len(rows)
            row_id = f"{meta_base['doc_id']}_utt_{str(utterance_index).zfill(6)}"
            page_values = [
                page
                for page in [record_start_page, *(item[1] for item in buffer)]
                if page is not None
            ]
            metadata = make_metadata(
                meta_base,
                "apollo_block",
                utterance_index,
                speaker=current_speaker,
                timestamp=current_timestamp,
                timestamp_sec=current_timestamp_sec,
                timestamp_valid=current_timestamp_valid,
                source_line_start=record_start_line or buffer[0][0],
                source_line_end=buffer[-1][0],
                page_start=min(page_values) if page_values else None,
                page_end=max(page_values) if page_values else None,
                speaker_label_status=current_speaker_status,
            )
            rows.append(make_row(row_id, utterance_text, metadata))
            buffer = []
            record_start_line = None
            record_start_page = None

        def begin_timestamp(
            timestamp: str,
            line_number: int,
            page_number: int | None,
        ) -> None:
            nonlocal current_timestamp, current_timestamp_sec
            nonlocal current_timestamp_valid, current_speaker
            nonlocal current_speaker_status, record_start_line
            nonlocal record_start_page, awaiting_speaker
            flush()
            current_timestamp = re.sub(r"\s+", " ", timestamp.strip())
            current_timestamp_sec, current_timestamp_valid = apollo_timestamp_info(
                current_timestamp
            )
            current_speaker = ""
            current_speaker_status = ""
            record_start_line = line_number
            record_start_page = page_number
            awaiting_speaker = True

        def begin_speaker(
            speaker: str,
            line_number: int,
            page_number: int | None,
            *,
            status: str = "recognized",
        ) -> None:
            nonlocal current_speaker, current_speaker_status
            nonlocal record_start_line, record_start_page, awaiting_speaker
            if current_speaker or buffer:
                flush()
            current_speaker = normalize_speaker(speaker)
            current_speaker_status = status
            if record_start_line is None:
                record_start_line = line_number
                record_start_page = page_number
            awaiting_speaker = False

        for line_number, page_number, raw_line in prepare_source_lines(
            text,
            apply_start_page=True,
        ):
            line = raw_line.strip()
            if not line:
                continue
            if is_common_artifact_line(line, remove_numeric_only_lines=False):
                continue

            if two_digit_re.fullmatch(line) and not current_speaker:
                pending_digits.append((line_number, page_number, line))
                if len(pending_digits) == 4:
                    begin_timestamp(
                        " ".join(item[2] for item in pending_digits),
                        pending_digits[0][0],
                        pending_digits[0][1],
                    )
                    pending_digits = []
                continue
            if pending_digits:
                pending_digits = []

            fused_match = spaced_fused_re.fullmatch(line)
            if fused_match:
                begin_timestamp(
                    fused_match.group("timestamp"),
                    line_number,
                    page_number,
                )
                begin_speaker(
                    fused_match.group("speaker"),
                    line_number,
                    page_number,
                    status=speaker_status(fused_match.group("speaker")),
                )
                if fused_match.group("text"):
                    buffer.append((line_number, page_number, fused_match.group("text")))
                continue

            no_space_match = no_space_fused_re.fullmatch(line)
            if no_space_match:
                begin_timestamp(
                    no_space_match.group("timestamp"),
                    line_number,
                    page_number,
                )
                begin_speaker(
                    no_space_match.group("speaker"),
                    line_number,
                    page_number,
                )
                buffer.append((line_number, page_number, no_space_match.group("text")))
                continue

            no_space_speaker_match = no_space_speaker_only_re.fullmatch(line)
            if no_space_speaker_match:
                begin_timestamp(
                    no_space_speaker_match.group("timestamp"),
                    line_number,
                    page_number,
                )
                begin_speaker(
                    no_space_speaker_match.group("speaker"),
                    line_number,
                    page_number,
                )
                continue

            timestamp_match = timestamp_re.fullmatch(line)
            if timestamp_match:
                begin_timestamp(
                    timestamp_match.group("timestamp"),
                    line_number,
                    page_number,
                )
                continue

            ocr_timestamp_match = ocr_timestamp_re.fullmatch(line)
            if ocr_timestamp_match:
                begin_timestamp(
                    ocr_timestamp_match.group("timestamp"),
                    line_number,
                    page_number,
                )
                continue

            president_match = president_with_text_re.fullmatch(line)
            if president_match:
                begin_speaker(
                    president_match.group("speaker"),
                    line_number,
                    page_number,
                )
                buffer.append((line_number, page_number, president_match.group("text")))
                continue

            speaker_match = speaker_line_re.fullmatch(line)
            if speaker_match:
                begin_speaker(
                    speaker_match.group(1),
                    line_number,
                    page_number,
                    status=speaker_status(speaker_match.group(1)),
                )
                continue

            if awaiting_speaker:
                joined_match = joined_speaker_re.fullmatch(line)
                if joined_match:
                    begin_speaker(
                        joined_match.group("speaker"),
                        line_number,
                        page_number,
                    )
                    buffer.append(
                        (line_number, page_number, joined_match.group("text"))
                    )
                    continue
                awaiting_match = awaiting_speaker_with_text_re.fullmatch(line)
                if awaiting_match:
                    begin_speaker(
                        awaiting_match.group("speaker"),
                        line_number,
                        page_number,
                        status=speaker_status(awaiting_match.group("speaker")),
                    )
                    buffer.append(
                        (line_number, page_number, awaiting_match.group("text"))
                    )
                    continue

            if current_speaker:
                buffer.append((line_number, page_number, line))
                continue
        flush()
        return rows

    def parse_pao(text: str, path: Path | None) -> list[dict[str, Any]]:
        meta_base = infer_doc_metadata(path)
        rows: list[dict[str, Any]] = []
        current_speaker = current_segment_marker = ""
        current_calendar_date = ""
        buffer: list[tuple[int, int | None, str]] = []
        record_start_line: int | None = None
        record_start_page: int | None = None
        at_boundary = True
        segment_re = re.compile(
            r"^\s*CDT\b.*GET\b.*$",
            flags=re.IGNORECASE,
        )
        calendar_date_re = re.compile(
            r"^\s*\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\s*$"
        )
        pao_header_re = re.compile(r"^\s*(APOLLO\s+\d+\s+MISSION\s+COMMENTARY.*|APOLLO\s+\d+\s+STATUS\s+REPORT.*|APOLLO\s+\d+\s+SPACECRAFT\s+COMMENTARY.*|CDT\b.*GET\b.*|MC\s+\d+)\s*$", flags=re.IGNORECASE)
        inline_speaker_names = [
            "PRESIDENT NIXON",
            "COMM TECH",
            "CAP COM",
            "CAPCOM",
            "HORNET",
            "SWIM 1",
            "MCC",
            "IWO",
        ]
        inline_speaker_pattern = "|".join(
            re.escape(name) for name in inline_speaker_names
        )
        inline_speaker_re = re.compile(
            rf"^(?P<speaker>{inline_speaker_pattern})"
            r"(?:\s*[:\-]\s*|\s+)(?P<text>\S.*)$"
        )

        def flush() -> None:
            nonlocal buffer, record_start_line, record_start_page
            if not buffer:
                return
            utterance_text = _clean_text("\n".join(item[2] for item in buffer))
            if not utterance_text:
                buffer = []
                record_start_line = None
                record_start_page = None
                return
            speaker = normalize_speaker(current_speaker or implicit_speaker or "PAO")
            utterance_index = len(rows)
            row_id = f"{meta_base['doc_id']}_utt_{str(utterance_index).zfill(6)}"
            page_values = [
                page
                for page in [record_start_page, *(item[1] for item in buffer)]
                if page is not None
            ]
            metadata = make_metadata(
                meta_base,
                "pao_commentary",
                utterance_index,
                speaker=speaker,
                segment_marker=current_segment_marker,
                source_line_start=record_start_line or buffer[0][0],
                source_line_end=buffer[-1][0],
                page_start=min(page_values) if page_values else None,
                page_end=max(page_values) if page_values else None,
                calendar_date=current_calendar_date,
            )
            rows.append(make_row(row_id, utterance_text, metadata))
            buffer = []
            record_start_line = None
            record_start_page = None

        for line_number, page_number, raw_line in prepare_source_lines(
            text,
            apply_start_page=True,
        ):
            line = raw_line.strip()
            if not line:
                flush()
                at_boundary = True
                continue
            if calendar_date_re.fullmatch(line):
                flush()
                current_calendar_date = line
                at_boundary = True
                continue
            if pao_header_re.fullmatch(line):
                flush()
                current_segment_marker = line
                current_speaker = implicit_speaker
                at_boundary = True
                continue
            if segment_re.fullmatch(line):
                flush()
                current_segment_marker = line
                current_speaker = implicit_speaker
                at_boundary = True
                continue
            if is_common_artifact_line(line, remove_numeric_only_lines=True):
                continue
            speaker_match = speaker_line_re.fullmatch(line)
            if speaker_match:
                flush()
                current_speaker = normalize_speaker(speaker_match.group(1))
                record_start_line = line_number
                record_start_page = page_number
                at_boundary = False
                continue
            if at_boundary:
                inline_match = inline_speaker_re.fullmatch(line)
                if inline_match:
                    flush()
                    current_speaker = normalize_speaker(inline_match.group("speaker"))
                    record_start_line = line_number
                    record_start_page = page_number
                    buffer.append((line_number, page_number, inline_match.group("text")))
                    at_boundary = False
                    continue
            if not current_speaker:
                current_speaker = implicit_speaker
            if record_start_line is None:
                record_start_line = line_number
                record_start_page = page_number
            buffer.append((line_number, page_number, line))
            at_boundary = False
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
            text = path.read_text(encoding="utf-8")
            selected = detect_parser(text, path)
            if selected == "challenger":
                all_rows.extend(parse_challenger(text, path))
            elif selected == "apollo":
                all_rows.extend(parse_apollo(text, path))
            elif selected == "pao":
                all_rows.extend(parse_pao(text, path))

    columns = ["id", "document", "metadata"]
    if not all_rows:
        df = pd.DataFrame(columns=columns)
        validate_cleaned_df(df)
        return df
    df = pd.DataFrame(all_rows).reindex(columns=columns)
    df = df[df["document"].astype(str).str.strip().ne("")].reset_index(drop=True)
    validate_cleaned_df(df)
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
        stop_heading="DISTRIBUTION:",
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
    validate_cleaned_df(all_report_df)
    validate_cleaned_df(all_transcript_df)
    validate_cleaned_df(all_nasa_df)
    return all_report_df, all_transcript_df, all_nasa_df


if __name__ == "__main__":
    reports, transcripts, all_data = build_all_nasa_dataframes("data_text")
    print(build_debug_df(all_data).head(20))
