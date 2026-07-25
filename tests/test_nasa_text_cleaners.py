import unittest
from pathlib import Path
import re

from chromadb.api.types import validate_metadatas
from embedding_pipeline import ChromaEmbeddingPipelineTextOnly
from nasa_text_cleaners import (
    build_all_nasa_dataframes,
    nasa_report_to_clean_df,
    nasa_transcripts_to_clean_df,
    validate_cleaned_df,
)


class ReportCleanerContractTests(unittest.TestCase):
    def clean_report(self, raw_text: str, **kwargs):
        return nasa_report_to_clean_df(
            raw_text=raw_text,
            source_file="fixture.txt",
            mission="Apollo 11",
            collection="apollo11",
            report_type="test_report",
            **kwargs,
        )

    @staticmethod
    def combined_text(df) -> str:
        return "\n".join(df["document"].tolist())

    def test_mixed_figure_page_preserves_prose_and_rejects_visual_body(self):
        raw = """--- PAGE 1 ---
1. OPERATIONS
The spacecraft continued on the planned trajectory while controllers monitored all primary systems.
Telemetry confirmed that propulsion, guidance, and electrical performance remained within expected limits.
FIGURE 2-1. FLIGHT PATH
GROUND TIME MINUS VEHICLE TIME, milliseconds
99 88 77 66 ++++ VISUAL-OCR-SENTINEL
10 | 20 | 30 | 40
The crew then completed the scheduled navigation check and reported stable spacecraft conditions.
Mission Control reviewed the new measurements and approved continuation of the planned sequence.
"""
        df = self.clean_report(raw)
        text = self.combined_text(df)

        self.assertIn("spacecraft continued on the planned trajectory", text)
        self.assertIn("crew then completed the scheduled navigation check", text)
        self.assertNotIn("VISUAL-OCR-SENTINEL", text)
        self.assertNotIn("GROUND TIME MINUS VEHICLE TIME", text)

    def test_pure_visual_page_is_rejected(self):
        raw = """--- PAGE 1 ---
FIGURE 4-2. ASCENT TRAJECTORY
99 88 77 66 ++++ VISUAL-OCR-SENTINEL
0 | 10 | 20 | 30 | 40
(225) (156) (23) (55) (42) 25k
35R
42R
SOR
"""
        df = self.clean_report(raw)

        self.assertTrue(df.empty)

    def test_does_not_merge_across_a_removed_visual_page(self):
        raw = """--- PAGE 1 ---
The first operating sequence remained incomplete because

--- PAGE 2 ---
FIGURE 9-9. SYSTEM MAP
99 88 77 66 ++++ VISUAL-OCR-SENTINEL
0 | 10 | 20 | 30 | 40
(225) (156) (23) (55) (42) 25k
35R
42R

--- PAGE 3 ---
the later anomaly review identified a separate wiring fault in the data unit.
"""
        df = self.clean_report(raw)
        spans = {
            (row["page_start"], row["page_end"])
            for row in df["metadata"]
        }

        self.assertNotIn((1, 3), spans)
        self.assertFalse(
            any(
                "first operating sequence" in document
                and "later anomaly review" in document
                for document in df["document"]
            )
        )

    def test_stop_heading_keeps_prefix_on_same_page(self):
        raw = """--- PAGE 1 ---
The final analysis confirmed that all primary mission objectives had been completed successfully.
The recovered flight data also supported the conclusions documented in this report.
REFERENCES
Smith, A. A document that should not enter the retrieval corpus.
"""
        df = self.clean_report(raw, stop_heading="REFERENCES")
        text = self.combined_text(df)

        self.assertIn("final analysis confirmed", text)
        self.assertNotIn("Smith, A.", text)

    def test_launch_table_date_is_not_a_heading_and_approval_text_is_retained(self):
        raw = """--- PAGE 1 ---
16 Jul 69 0932 EDT
39A
10.203

--- PAGE 2 ---
APPENDIX B
APPROVAL
The information in this report has been reviewed for security classification.
The highest classification has been determined to be unclassified.
"""
        df = self.clean_report(raw)

        self.assertEqual(len(df), 1)
        self.assertEqual(
            df.iloc[0]["metadata"]["section_path"],
            "APPENDIX B > APPROVAL",
        )
        self.assertIn("determined to be unclassified", df.iloc[0]["document"])
        self.assertNotIn("16. Jul 69 0932 EDT", df.iloc[0]["document"])

    def test_intentionally_blank_marker_is_removed_without_losing_neighbors(self):
        raw = """--- PAGE 1 ---
1. OPERATIONS
The crew completed the first planned navigation check and reported stable spacecraft conditions.

--- PAGE 2 ---
) ) ) ) THIS PAGE INTENTIONALLY LEFT BLANK.

--- PAGE 3 ---
Controllers then reviewed the measurements and approved the next scheduled mission activity.
"""
        df = self.clean_report(raw)
        text = self.combined_text(df)

        self.assertIn("first planned navigation check", text)
        self.assertIn("Controllers then reviewed", text)
        self.assertNotIn("INTENTIONALLY LEFT BLANK", text)

    def test_visual_axis_label_does_not_split_or_relabel_prose(self):
        raw = """--- PAGE 1 ---
11. THE LUNAR SURFACE
Passive Seismic Experiment
The signals have emergent onsets and last up to seven minutes for the largest
Frequency. Hz
RMS amplitude. arbitrary units
of these trains observed by the long-period vertical-component seismometer.
"""
        df = self.clean_report(raw)
        text = self.combined_text(df)

        self.assertIn("largest of these trains", text)
        self.assertNotIn("Frequency. Hz", text)
        self.assertNotIn("RMS amplitude", text)
        self.assertFalse(
            any(
                "Frequency. Hz" in metadata["section_path"]
                for metadata in df["metadata"]
            )
        )

    def test_interleaved_figure_caption_does_not_break_narrative_sentence(self):
        raw = """--- PAGE 1 ---
11. THE LUNAR SURFACE
Passive Seismic Experiment
The signals have emergent onsets and last up to seven minutes for the largest
Frequency. Hz
of these trains are also observed on the
Figure 11-24.- Seismometer response from
seismograms from the long-period vertical-
first portable life support system
component seismometer. As shown in
impacting lunar surface.
figure 11-25, the events associated with
these signals began two days before lunar noon and continued afterward.
"""
        df = self.clean_report(raw)
        text = self.combined_text(df)

        self.assertIn(
            "largest of these trains are also observed on the "
            "seismograms from the long-period verticalcomponent seismometer. "
            "As shown in figure 11-25, the events associated with these signals began",
            text,
        )
        self.assertNotIn("impacting lunar surface", text)

    def test_mixed_page_preserves_top_level_section_reset(self):
        raw = """--- PAGE 1 ---
3. LAUNCH OPERATIONS
The launch operations team completed the planned checkout and verified all required support systems.

--- PAGE 2 ---
SECTION 4
TRAJECTORY
4.1 SUMMARY
The trajectory parameters from launch through translunar injection remained close to the nominal plan.
Flight controllers reviewed the tracking data and confirmed the expected vehicle performance.
FIGURE 4-1. ASCENT TRAJECTORY
0 | 10 | 20 | 30
"""
        df = self.clean_report(raw)
        row = df[
            df["document"].str.contains("trajectory parameters", case=False)
        ].iloc[0]

        self.assertEqual(row["metadata"]["top_section_id"], "4")
        self.assertEqual(
            row["metadata"]["section_path"],
            "4. TRAJECTORY > 4.1. SUMMARY",
        )


class TranscriptCleanerContractTests(unittest.TestCase):
    def clean_apollo(self, raw_text: str, **kwargs):
        return nasa_transcripts_to_clean_df(
            raw_text=raw_text,
            parser="apollo",
            source_file="apollo_fixture.txt",
            mission="Apollo 11",
            collection="apollo11",
            **kwargs,
        )

    def clean_pao(self, raw_text: str, **kwargs):
        return nasa_transcripts_to_clean_df(
            raw_text=raw_text,
            parser="pao",
            source_file="pao_fixture.txt",
            mission="Apollo 11",
            collection="apollo11",
            **kwargs,
        )

    def test_annotated_speakers_preserve_historic_dialogue(self):
        raw = """--- PAGE 11 ---
00 01 00 00
CDR (EAGLE)
Houston, Tranquility Base here. The Eagle has landed.
00 01 00 05
LMP (EVA)
Very smooth touchdown.
00 01 00 10
CMP (COLUMBIA)
The Eagle has wings.
"""
        df = self.clean_apollo(raw, start_page=11)

        self.assertEqual(
            df["metadata"].map(lambda value: value["speaker"]).tolist(),
            ["CDR (EAGLE)", "LMP (EVA)", "CMP (COLUMBIA)"],
        )
        self.assertIn(
            "Houston, Tranquility Base here. The Eagle has landed.",
            df["document"].tolist(),
        )

    def test_fused_timestamp_speaker_line_is_recovered(self):
        raw = "00 01 18 42LMPOkay, go ahead and talk."
        df = self.clean_apollo(raw)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["document"], "Okay, go ahead and talk.")
        self.assertEqual(df.iloc[0]["metadata"]["speaker"], "LMP")
        self.assertEqual(df.iloc[0]["metadata"]["timestamp"], "00 01 18 42")
        self.assertEqual(df.iloc[0]["metadata"]["timestamp_sec"], 4722)

    def test_timestamp_sign_validation_and_original_line_provenance(self):
        raw = """ignored cover line
--- PAGE 11 ---
-00 00 00 10
CDR
Launch is ten seconds away.
00 93 00 00
LMP
This timestamp contains an OCR error but the dialogue must remain.
"""
        df = self.clean_apollo(raw, start_page=11)
        first_meta = df.iloc[0]["metadata"]
        second_meta = df.iloc[1]["metadata"]

        self.assertEqual(first_meta["timestamp_sec"], -10)
        self.assertTrue(first_meta["timestamp_valid"])
        self.assertEqual(first_meta["source_line_start"], 3)
        self.assertEqual(first_meta["source_line_end"], 5)
        self.assertFalse(second_meta["timestamp_valid"])
        self.assertNotIn("timestamp_sec", second_meta)
        self.assertIn("OCR error", df.iloc[1]["document"])

    def test_missing_requested_start_page_raises_clear_error(self):
        raw = """--- PAGE 1 ---
00 00 00 01
CDR
Test dialogue.
"""
        with self.assertRaisesRegex(ValueError, "start page 99"):
            self.clean_apollo(raw, start_page=99)

    def test_split_tape_page_header_is_removed_without_losing_dialogue(self):
        raw = """00 01 00 00
CDR
You're right.
Tape
7/8
-
Page
53
END
OF
TAPE
<<<<
Continue with the checklist.
"""
        df = self.clean_apollo(raw)

        self.assertEqual(len(df), 1)
        self.assertEqual(
            df.iloc[0]["document"],
            "You're right. Continue with the checklist.",
        )
        self.assertNotRegex(df.iloc[0]["document"], r"(?i)Tape|Page|END OF")
        self.assertEqual(df.iloc[0]["metadata"]["source_line_end"], 13)

    def test_ocr_corrupted_voice_titles_are_removed_by_tape_context(self):
        title_variants = [
            "A CLO 13 AIR-TO-GROUND VOICE TRAN. RIPTION",
            "AP\n0 13 AIR-TO-GROUND VOICE TRANS .PTION",
            "APOLLO 13 AIR-TO-GROUND VOICE THANSCRIPTION",
            "A LO 13 AIR-TO-GROUND VOICE TRAN (IPTION",
            "APOLLO 13 AIR-TO-GROUND VOICE MASSCRIPPION",
            "APOLLO 13 AIR-TO-GROUND VOICE PRABSCRIPTION",
        ]
        for title in title_variants:
            with self.subTest(title=title):
                raw = f"""--- PAGE 417 ---
{title}
Tape 61/1
Page 410
03 17 58 23
CC
Aquarius, Houston. Over.
"""
                df = self.clean_apollo(raw)

                self.assertEqual(
                    df["document"].tolist(),
                    ["Aquarius, Houston. Over."],
                )

    def test_corrupted_page_label_and_timestamp_preserve_boundary(self):
        raw = """05 08 01 32
CC
Hey, Jim, no further correction is required at this time.
END OF TAPE

--- PAGE 671 ---
APOLLO 13 AIR-TO-GROUND VOICE PRABSCRIPTION
Tape 86/1
Payte only
0% 08 01 10
CMP
Yes. We are ready to continue.

--- PAGE 672 ---
APOLLO 13 AIR-TO-GROUND VOICE MASSCRIPPION
Tape 73/1
Pragre 530
05 08 02 00
CC
Copy that.
"""
        df = self.clean_apollo(raw)
        first_meta, second_meta, third_meta = df["metadata"].tolist()

        self.assertEqual(
            df["document"].tolist(),
            [
                "Hey, Jim, no further correction is required at this time.",
                "Yes. We are ready to continue.",
                "Copy that.",
            ],
        )
        self.assertEqual(first_meta["timestamp"], "05 08 01 32")
        self.assertTrue(first_meta["timestamp_valid"])
        self.assertEqual(second_meta["timestamp"], "0% 08 01 10")
        self.assertFalse(second_meta["timestamp_valid"])
        self.assertNotIn("timestamp_sec", second_meta)
        self.assertEqual(third_meta["timestamp"], "05 08 02 00")
        self.assertTrue(third_meta["timestamp_valid"])

    def test_pao_date_preserves_speaker_and_get_segment(self):
        raw = """CDT 10:00 GET 001:00
CAPCOM
First instruction.
7-20-69
Second instruction.
SC
MS mode to null 1.
"""
        df = self.clean_pao(raw)
        first_meta, second_meta, third_meta = df["metadata"].tolist()

        self.assertEqual(
            [first_meta["speaker"], second_meta["speaker"], third_meta["speaker"]],
            ["CAPCOM", "CAPCOM", "SC"],
        )
        self.assertEqual(
            first_meta["segment_marker"],
            second_meta["segment_marker"],
        )
        self.assertEqual(second_meta["calendar_date"], "7-20-69")
        self.assertEqual(df.iloc[2]["document"], "MS mode to null 1.")

    def test_additional_fused_and_uncertain_speaker_forms_are_retained(self):
        raw = """00 01 59 40CDR ... the time, is that right?
03 08 08 46LMPI like the neat way he's got his safety belt on.
03 08 09 05CDR
This speaker-only fused record is retained.
04 14 16 30 PRESIDENT NIXON Neil and Buzz, I am talking to you.
PRESIDENT NIXON And thank you very much.
05 07 03 11
CMI'
This OCR speaker label is uncertain, but its dialogue remains.
06 00 00 01
HORNET 11, Hornet. Copy the recovery data.
06 00 00 02
CMP/CDR Go ahead.
06 00 00 03
CMPMARK.
06 00 00 04
CDR/CMP
Yes, they're OFF.
"""
        df = self.clean_apollo(raw)
        text = "\n".join(df["document"])

        self.assertIn("the time, is that right?", text)
        self.assertIn("I like the neat way", text)
        self.assertIn("speaker-only fused record", text)
        self.assertIn("Neil and Buzz", text)
        self.assertIn("And thank you very much.", text)
        self.assertIn("11, Hornet. Copy the recovery data.", text)
        self.assertIn("Go ahead.", text)
        self.assertIn("MARK.", text)
        self.assertIn("Yes, they're OFF.", text)
        alias_row = df[
            df["metadata"].map(lambda value: value["speaker"] == "CMI'")
        ].iloc[0]
        self.assertEqual(
            alias_row["metadata"]["speaker_label_status"],
            "ocr_alias",
        )

    def test_four_line_timestamp_after_active_speaker_is_reconstructed(self):
        raw = """00 00 31 06
CDR
First statement.
00
00
31
09
CMP
Yes.
"""
        df = self.clean_apollo(raw)

        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]["metadata"]["speaker"], "CMP")
        self.assertEqual(df.iloc[1]["metadata"]["timestamp"], "00 00 31 09")
        self.assertEqual(df.iloc[1]["metadata"]["timestamp_sec"], 1869)
        self.assertEqual(df.iloc[1]["document"], "Yes.")


class CleanerEmbeddingCompatibilityTests(unittest.TestCase):
    def test_cleaner_source_units_generate_unique_pipeline_ids(self):
        raw = """--- PAGE 1 ---
The first complete report statement contains useful operational evidence for retrieval.

The second complete report statement contains different mission evidence for retrieval.
"""
        df = nasa_report_to_clean_df(
            raw_text=raw,
            source_file="fixture.txt",
            mission="Apollo 11",
            collection="apollo11",
            report_type="test_report",
        )
        self.assertEqual(len(df), 2)

        pipeline = object.__new__(ChromaEmbeddingPipelineTextOnly)
        pipeline.chunk_size = 1000
        pipeline.chunk_overlap = 100
        generated_ids = []
        for _, row in df.iterrows():
            for _text, metadata in pipeline.chunk_text(
                row["document"],
                row["metadata"],
            ):
                generated_ids.append(
                    pipeline.generate_document_id(Path("fixture.txt"), metadata)
                )

        self.assertEqual(len(generated_ids), len(set(generated_ids)))


DATA_DIR = Path(__file__).resolve().parents[1] / "data_text"


@unittest.skipUnless(DATA_DIR.exists(), "NASA corpus is not available")
class CorpusInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports, cls.transcripts, cls.all_data = build_all_nasa_dataframes(
            DATA_DIR
        )

    def test_all_expected_sources_have_clean_records(self):
        expected_sources = {
            "19900066485_textract_full_text.txt",
            "Apollo_11_Flight_Plan_HSK_textract_full_text.txt",
            "NASA_NTRS_Archive_19710015566_textract_full_text.txt",
            "107-AAG_STS-51L_Mission_Audio_transcript.txt",
            "108-AAG_STS-51L_Mission_Audio_transcript.txt",
            "109-AAG_STS-51L_Mission_Audio_transcript.txt",
            "a11transcript_pao_textract_full_text.txt",
            "a11transcript_tec_textract_full_text.txt",
            "a11transscript_cm_textract_full_text.txt",
            "AS13_PAO_textract_full_text.txt",
            "AS13_TEC_textract_full_text.txt",
            "AS13_CM_textract_full_text.txt",
        }
        actual_sources = {
            metadata["source_file"] for metadata in self.all_data["metadata"]
        }

        self.assertEqual(actual_sources, expected_sources)

    def test_cleaned_metadata_is_valid_for_pipeline_and_chroma(self):
        validate_cleaned_df(self.all_data)
        validate_metadatas(self.all_data["metadata"].tolist())
        sources = self.all_data["metadata"].map(lambda value: value["source"])

        self.assertFalse(sources.duplicated().any())
        self.assertTrue(
            all(
                Path(metadata["source_path"]).is_file()
                for metadata in self.all_data["metadata"]
            )
        )

    def test_report_section_paths_do_not_mix_top_level_numbers(self):
        for metadata in self.reports["metadata"]:
            section_numbers = re.findall(
                r"(?:^| > )(\d+(?:\.\d+)*)\.",
                metadata["section_path"],
            )
            roots = {number.split(".")[0] for number in section_numbers}
            self.assertLessEqual(
                len(roots),
                1,
                msg=metadata["section_path"],
            )

    def test_known_report_prose_is_retained_and_visual_noise_is_excluded(self):
        report_text = "\n".join(self.reports["document"])
        for phrase in [
            "Nominal launch time is 9:32 EDT",
            "Passive thermal control",
            "released from quarantine on August 10, 1969",
            "The predicted times for establishing actual minus predicted times",
            "Tracking data from seven C-Band radar stations",
        ]:
            self.assertIn(phrase.casefold(), report_text.casefold())
        for artifact in [
            "(225) (156) (23) (55) (42) 25k",
            "Office of the Asst. Sec. of Defense",
            "GROUND TIME MINUS VEHICLE TIME",
        ]:
            self.assertNotIn(artifact.casefold(), report_text.casefold())

        self.assertFalse(
            any(
                "intermittent continuity to the automatic coils" in document
                and "data unit results in the undesirable" in document
                for document in self.reports["document"]
            )
        )

    def test_historic_apollo_dialogue_is_retained_without_page_headers(self):
        transcript_text = "\n".join(self.transcripts["document"])
        for phrase in [
            "Eagle is undocked",
            "The Eagle has wings",
            "Houston, Tranquility Base here",
            "THE EAGLE HAS LANDED",
            "Very smooth touchdown",
        ]:
            self.assertIn(phrase.casefold(), transcript_text.casefold())

        self.assertIsNone(
            re.search(
                r"(?is)\bTape\s+\d+/\d+.*?(?:Page|Fage)\s+\d+\b",
                transcript_text,
            )
        )
        self.assertIsNone(re.search(r"(?i)END\s+OF\s+TAPE", transcript_text))
        self.assertIsNone(
            re.search(
                r"(?i)(?:(?:AIR|ATR)-TO-(?:GROUND|GROUID)|ONBOARD)"
                r"\s+VOICE\s+(?:TRANSCRIPTION|TRANCCRIPTION)",
                transcript_text,
            )
        )
        self.assertIsNone(re.search(r"(?i)GOSS\s+NET", transcript_text))
        self.assertIsNone(re.search(r"(?i)\bA\s+POLLO\s+\d+", transcript_text))
        self.assertIsNone(re.search(r"(?i)\bAPOLLO\s+B\b", transcript_text))
        invalid_timestamps = [
            metadata
            for metadata in self.transcripts["metadata"]
            if metadata.get("timestamp_valid") is False
        ]
        self.assertTrue(invalid_timestamps)
        self.assertTrue(
            all("timestamp_sec" not in metadata for metadata in invalid_timestamps)
        )


if __name__ == "__main__":
    unittest.main()
