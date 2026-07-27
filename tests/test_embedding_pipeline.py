import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from embedding_pipeline import ChromaEmbeddingPipelineTextOnly


class CleanedRecordAggregationTests(unittest.TestCase):
    def test_aggregates_transcript_turns_in_source_order(
        self,
    ):
        pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        source_path = Path(
            "data_text/apollo11/transcript.txt"
        )
        common_metadata = {
            "collection": "apollo11",
            "mission": "Apollo 11",
            "source_type": "transcript",
            "source_file": source_path.name,
            "source_path": str(source_path),
            "doc_id": "apollo11_transcript",
            "timestamp_valid": True,
        }
        cleaned_df = pd.DataFrame(
            [
                {
                    "id": "turn-1",
                    "document": "Roger. Proceed.",
                    "metadata": {
                        **common_metadata,
                        "source": "turn-1",
                        "utterance_index": 1,
                        "speaker": "CDR",
                        "timestamp": "000 00 00 20",
                        "timestamp_sec": 20,
                        "source_line_start": 12,
                        "source_line_end": 13,
                    },
                },
                {
                    "id": "turn-0",
                    "document": "You are go for landing.",
                    "metadata": {
                        **common_metadata,
                        "source": "turn-0",
                        "utterance_index": 0,
                        "speaker": "CC",
                        "timestamp": "000 00 00 10",
                        "timestamp_sec": 10,
                        "source_line_start": 10,
                        "source_line_end": 11,
                    },
                },
            ]
        )

        aggregated = (
            pipeline.aggregate_cleaned_records_by_file(
                cleaned_df
            )
        )

        self.assertEqual(list(aggregated), [source_path])
        text, metadata = aggregated[source_path]
        self.assertLess(
            text.index("You are go for landing."),
            text.index("Roger. Proceed."),
        )
        self.assertIn("utt=000000", text)
        self.assertIn("time=000 00 00 10", text)
        self.assertIn("speaker=CC", text)
        self.assertIn("lines=10-11", text)
        self.assertIn("utt=000001", text)
        self.assertIn("speaker=CDR", text)
        self.assertEqual(
            metadata,
            {
                "collection": "apollo11",
                "mission": "Apollo 11",
                "source_type": "transcript",
                "source_file": source_path.name,
                "source_path": str(source_path),
                "doc_id": "apollo11_transcript",
                "source": "apollo11_transcript",
                "record_count": 2,
                "source_line_start": 10,
                "source_line_end": 13,
            },
        )

    def test_aggregates_report_blocks_with_page_provenance(
        self,
    ):
        pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        source_path = Path(
            "data_text/apollo13/mission_report.txt"
        )
        common_metadata = {
            "collection": "apollo13",
            "mission": "Apollo 13",
            "source_type": "report",
            "source_file": source_path.name,
            "source_path": str(source_path),
            "doc_id": "apollo13_mission_report",
            "report_type": "mission_report",
        }
        cleaned_df = pd.DataFrame(
            [
                {
                    "id": "report-0",
                    "document": (
                        "1.0 Mission Overview\n\n"
                        "Apollo 13 launched on April 11, 1970."
                    ),
                    "metadata": {
                        **common_metadata,
                        "source": "report-0",
                        "section_path": "1.0 Mission Overview",
                        "page_start": 3,
                        "page_end": 3,
                    },
                },
                {
                    "id": "report-1",
                    "document": (
                        "2.0 Anomaly\n\n"
                        "An oxygen tank failure changed the mission."
                    ),
                    "metadata": {
                        **common_metadata,
                        "source": "report-1",
                        "section_path": "2.0 Anomaly",
                        "page_start": 8,
                        "page_end": 9,
                    },
                },
            ]
        )

        aggregated = (
            pipeline.aggregate_cleaned_records_by_file(
                cleaned_df
            )
        )

        self.assertEqual(list(aggregated), [source_path])
        text, metadata = aggregated[source_path]
        self.assertLess(
            text.index("1.0 Mission Overview"),
            text.index("2.0 Anomaly"),
        )
        self.assertIn("block=00000", text)
        self.assertIn("pages=3-3", text)
        self.assertIn("block=00001", text)
        self.assertIn("pages=8-9", text)
        self.assertIn(
            "Apollo 13 launched on April 11, 1970.",
            text,
        )
        self.assertIn(
            "An oxygen tank failure changed the mission.",
            text,
        )
        self.assertEqual(
            metadata,
            {
                "collection": "apollo13",
                "mission": "Apollo 13",
                "source_type": "report",
                "source_file": source_path.name,
                "source_path": str(source_path),
                "doc_id": "apollo13_mission_report",
                "source": "apollo13_mission_report",
                "record_count": 2,
                "page_start": 3,
                "page_end": 9,
            },
        )

    def test_preserves_invalid_transcript_timestamp(
        self,
    ):
        pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        source_path = Path(
            "data_text/apollo13/transcript.txt"
        )
        cleaned_df = pd.DataFrame(
            [
                {
                    "id": "turn-invalid-time",
                    "document": (
                        "The timestamp contains an OCR error."
                    ),
                    "metadata": {
                        "collection": "apollo13",
                        "mission": "Apollo 13",
                        "source_type": "transcript",
                        "source_file": source_path.name,
                        "source_path": str(source_path),
                        "doc_id": "apollo13_transcript",
                        "source": "turn-invalid-time",
                        "utterance_index": 0,
                        "speaker": "CC",
                        "timestamp": "OCR 55 99 99",
                        "timestamp_valid": False,
                        "source_line_start": 40,
                        "source_line_end": 41,
                    },
                },
            ]
        )

        aggregated = (
            pipeline.aggregate_cleaned_records_by_file(
                cleaned_df
            )
        )

        text, metadata = aggregated[source_path]
        self.assertIn("time=OCR 55 99 99", text)
        self.assertIn("time_valid=false", text)
        self.assertNotIn("timestamp_sec=", text)
        self.assertNotIn("timestamp_sec", metadata)

    def test_chunks_each_aggregated_file_once(self):
        pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        transcript_path = Path(
            "data_text/apollo11/transcript.txt"
        )
        report_path = Path(
            "data_text/apollo13/mission_report.txt"
        )
        transcript_metadata = {
            "source_type": "transcript",
            "source_path": str(transcript_path),
        }
        report_metadata = {
            "source_type": "report",
            "source_path": str(report_path),
        }
        cleaned_df = pd.DataFrame()
        pipeline.aggregate_cleaned_records_by_file = (
            MagicMock(
                return_value={
                    transcript_path: (
                        "aggregated transcript",
                        transcript_metadata,
                    ),
                    report_path: (
                        "aggregated report",
                        report_metadata,
                    ),
                }
            )
        )
        pipeline.chunk_text = MagicMock(
            side_effect=lambda text, metadata: [
                (
                    f"chunked: {text}",
                    metadata.copy(),
                )
            ]
        )

        documents_by_file = (
            pipeline.chunk_cleaned_records_by_file(
                cleaned_df
            )
        )

        pipeline.aggregate_cleaned_records_by_file.assert_called_once_with(
            cleaned_df
        )
        self.assertEqual(pipeline.chunk_text.call_count, 2)
        pipeline.chunk_text.assert_any_call(
            "aggregated transcript",
            transcript_metadata,
        )
        pipeline.chunk_text.assert_any_call(
            "aggregated report",
            report_metadata,
        )
        self.assertEqual(
            documents_by_file,
            {
                transcript_path: [
                    (
                        "chunked: aggregated transcript",
                        transcript_metadata,
                    )
                ],
                report_path: [
                    (
                        "chunked: aggregated report",
                        report_metadata,
                    )
                ],
            },
        )


class ChunkingRubricTests(unittest.TestCase):
    def test_runtime_chunk_settings_and_size_limit(self):
        text = " ".join(
            f"Sentence {i} describes a NASA mission event "
            "with useful technical context."
            for i in range(1, 25)
        )

        for chunk_size, chunk_overlap in (
            (64, 8),
            (128, 16),
        ):
            with self.subTest(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ):
                pipeline = object.__new__(
                    ChromaEmbeddingPipelineTextOnly
                )
                pipeline.chunk_size = chunk_size
                pipeline.chunk_overlap = chunk_overlap

                chunks = pipeline.chunk_text(
                    text,
                    {
                        "source_type": "report",
                        "mission": "apollo11",
                    },
                )

                self.assertGreater(len(chunks), 1)

                for _, metadata in chunks:
                    self.assertLessEqual(
                        metadata["token_count"],
                        chunk_size,
                    )

    @patch("embedding_pipeline.TokenTextSplitter")
    def test_forwards_configured_overlap_to_splitter(
        self,
        mocked_splitter_class,
    ):
        pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        pipeline.chunk_size = 32
        pipeline.chunk_overlap = 7

        text = " ".join(
            f"mission-event-{i}"
            for i in range(100)
        )
        mocked_splitter = mocked_splitter_class.return_value
        mocked_splitter.split_text.return_value = [
            "first overlapping chunk",
            "second overlapping chunk",
        ]

        chunks = pipeline.chunk_text(
            text,
            {
                "source_type": "report",
                "mission": "apollo11",
            },
        )

        mocked_splitter_class.assert_called_once_with(
            chunk_size=32,
            chunk_overlap=7,
            separator=" ",
            backup_separators=["\n\n"],
            keep_whitespaces=True,
        )
        mocked_splitter.split_text.assert_called_once_with(text)
        self.assertEqual(len(chunks), 2)


class CollectionMetadataRubricTests(unittest.TestCase):
    def test_stores_required_chunk_metadata(self):
        pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        pipeline.collection = MagicMock()
        pipeline.collection.get.return_value = {"ids": []}

        file_path = Path(
            "data_text/apollo11/"
            "NASA_NTRS_Archive_19710015566_textract_full_text.txt"
        )
        source = "mission_report_report_00000"

        stats = pipeline.add_documents_to_collection(
            documents=[
                (
                    "Apollo 11 mission report content.",
                    {
                        "collection": "apollo11",
                        "mission": "Apollo 11",
                        "source": source,
                        "source_file": file_path.name,
                        "source_path": str(file_path),
                        "source_type": "report",
                        "chunk_index": 0,
                    },
                )
            ],
            file_path=file_path,
            batch_size=10,
            update_mode="skip",
        )

        stored_metadata = (
            pipeline.collection.add.call_args.kwargs[
                "metadatas"
            ][0]
        )

        self.assertEqual(
            stored_metadata["mission"],
            "apollo11",
        )
        self.assertEqual(
            stored_metadata["source"],
            source,
        )
        self.assertEqual(
            stored_metadata["filepath"],
            str(file_path),
        )
        self.assertEqual(stats["added"], 1)


class UpdateModeRubricTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        self.pipeline.collection = MagicMock()
        self.file_path = Path(
            "data_text/apollo11/"
            "NASA_NTRS_Archive_19710015566_textract_full_text.txt"
        )
        self.metadata = {
            "collection": "apollo11",
            "mission": "Apollo 11",
            "source": "mission_report_report_00000",
            "source_file": self.file_path.name,
            "source_path": str(self.file_path),
            "source_type": "report",
            "chunk_index": 0,
        }
        self.documents = [
            (
                "Apollo 11 mission report content.",
                self.metadata,
            )
        ]
        self.document_id = self.pipeline.generate_document_id(
            self.file_path,
            {
                **self.metadata,
                "mission": "apollo11",
            },
        )

    def test_skip_mode_leaves_existing_document_unchanged(self):
        self.pipeline.collection.get.return_value = {
            "ids": [self.document_id]
        }

        stats = self.pipeline.add_documents_to_collection(
            documents=self.documents,
            file_path=self.file_path,
            update_mode="skip",
        )

        self.pipeline.collection.add.assert_not_called()
        self.pipeline.collection.update.assert_not_called()
        self.pipeline.collection.upsert.assert_not_called()
        self.assertEqual(
            stats,
            {
                "added": 0,
                "updated": 0,
                "skipped": 1,
            },
        )

    def test_update_mode_updates_existing_document(self):
        self.pipeline.collection.get.return_value = {
            "ids": [self.document_id]
        }

        stats = self.pipeline.add_documents_to_collection(
            documents=self.documents,
            file_path=self.file_path,
            update_mode="update",
        )

        self.pipeline.collection.update.assert_called_once()
        self.pipeline.collection.add.assert_not_called()
        self.pipeline.collection.upsert.assert_not_called()
        self.assertEqual(
            stats,
            {
                "added": 0,
                "updated": 1,
                "skipped": 0,
            },
        )

    def test_replace_mode_removes_stale_file_chunks(self):
        stale_document_id = (
            "apollo11::mission_report_report_00000::"
            "chunk_0001"
        )
        self.pipeline.collection.get.side_effect = [
            {"ids": [self.document_id]},
            {
                "ids": [
                    self.document_id,
                    stale_document_id,
                ]
            },
        ]

        stats = self.pipeline.add_documents_to_collection(
            documents=self.documents,
            file_path=self.file_path,
            update_mode="replace",
        )

        self.pipeline.collection.upsert.assert_called_once()
        self.pipeline.collection.delete.assert_called_once_with(
            ids=[stale_document_id]
        )
        self.assertEqual(
            stats,
            {
                "added": 0,
                "updated": 1,
                "skipped": 0,
            },
        )


class PersistenceAndStatsRubricTests(unittest.TestCase):
    @patch("embedding_pipeline.chromadb.PersistentClient")
    @patch("embedding_pipeline.OpenAIEmbeddingFunction")
    @patch("embedding_pipeline.OpenAI")
    def test_uses_configured_chroma_path_and_collection_name(
        self,
        mocked_openai,
        mocked_embedding_function_class,
        mocked_persistent_client_class,
    ):
        mocked_client = mocked_persistent_client_class.return_value
        mocked_collection = MagicMock()
        mocked_client.get_or_create_collection.return_value = (
            mocked_collection
        )

        pipeline = ChromaEmbeddingPipelineTextOnly(
            openai_api_key="test-key",
            openai_base_url="https://example.test/v1",
            chroma_persist_directory="rubric_chroma",
            collection_name="rubric_collection",
            embedding_model="test-embedding-model",
        )

        mocked_persistent_client_class.assert_called_once_with(
            path="rubric_chroma"
        )
        mocked_client.get_or_create_collection.assert_called_once_with(
            name="rubric_collection",
            embedding_function=(
                mocked_embedding_function_class.return_value
            ),
            metadata={"hnsw:space": "cosine"},
        )
        self.assertIs(
            pipeline.collection,
            mocked_collection,
        )
        mocked_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.test/v1",
        )

    def test_collection_stats_include_size_and_aggregates(self):
        pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        pipeline.collection = MagicMock()
        pipeline.collection.name = "nasa_space_missions_text"
        pipeline.collection.count.return_value = 3
        pipeline.collection.get.return_value = {
            "metadatas": [
                {
                    "mission": "apollo11",
                    "filepath": "data_text/apollo11/report.txt",
                    "source_type": "report",
                },
                {
                    "mission": "apollo11",
                    "filepath": "data_text/apollo11/report.txt",
                    "source_type": "report",
                },
                {
                    "mission": "challenger",
                    "filepath": (
                        "data_text/challenger/transcript.txt"
                    ),
                    "source_type": "transcript",
                },
            ]
        }

        stats = pipeline.get_collection_stats()

        self.assertEqual(stats["total_documents"], 3)
        self.assertEqual(stats["source_files"], 2)
        self.assertEqual(
            stats["missions"],
            {
                "apollo11": 2,
                "challenger": 1,
            },
        )
        self.assertEqual(
            stats["data_types"],
            {
                "report": 2,
                "transcript": 1,
            },
        )


class ProcessAllTextDataTests(unittest.TestCase):
    def test_aggregates_results_and_forwards_batch_size(self):
        pipeline = object.__new__(
            ChromaEmbeddingPipelineTextOnly
        )
        cleaned_sentinel = object()

        documents_by_file = {
            Path("data_text/apollo11/a.txt"): [
                ("Apollo 11 chunk", {"mission": "Apollo 11"}),
            ],
            Path("data_text/apollo13/b.txt"): [
                ("Apollo 13 chunk one", {"mission": "Apollo 13"}),
                ("Apollo 13 chunk two", {"mission": "Apollo 13"}),
            ],
        }

        def fake_chunking(cleaned_df):
            self.assertIs(cleaned_df, cleaned_sentinel)
            return documents_by_file

        received_calls = []

        def fake_add(
            documents,
            file_path,
            batch_size,
            update_mode,
        ):
            received_calls.append(
                (file_path, batch_size, update_mode)
            )

            if file_path.name == "a.txt":
                return {
                    "added": 1,
                    "updated": 0,
                    "skipped": 0,
                }

            return {
                "added": 1,
                "updated": 1,
                "skipped": 0,
            }

        pipeline.chunk_cleaned_records_by_file = fake_chunking
        pipeline.add_documents_to_collection = fake_add

        with patch(
            "embedding_pipeline.build_all_nasa_dataframes",
            return_value=(None, None, cleaned_sentinel),
        ) as mocked_builder:
            stats = pipeline.process_all_text_data(
                "data_text",
                update_mode="update",
                batch_size=17,
            )

        mocked_builder.assert_called_once_with("data_text")

        self.assertEqual(stats["files_processed"], 2)
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["documents_added"], 2)
        self.assertEqual(stats["documents_updated"], 1)
        self.assertEqual(stats["documents_skipped"], 0)
        self.assertEqual(stats["errors"], 0)

        self.assertEqual(
            [call[1] for call in received_calls],
            [17, 17],
        )
        self.assertEqual(
            [call[2] for call in received_calls],
            ["update", "update"],
        )

        self.assertEqual(
            stats["missions"]["Apollo 11"]["chunks"],
            1,
        )
        self.assertEqual(
            stats["missions"]["Apollo 13"]["chunks"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
