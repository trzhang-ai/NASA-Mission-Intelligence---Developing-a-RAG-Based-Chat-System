import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from embedding_pipeline import ChromaEmbeddingPipelineTextOnly


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
