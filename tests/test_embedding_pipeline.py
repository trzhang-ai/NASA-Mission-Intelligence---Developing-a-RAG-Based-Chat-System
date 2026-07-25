import unittest
from pathlib import Path
from unittest.mock import patch

from embedding_pipeline import ChromaEmbeddingPipelineTextOnly


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
