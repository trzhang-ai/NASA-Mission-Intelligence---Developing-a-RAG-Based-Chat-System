import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_client import (
    discover_chroma_backends,
    format_context,
    initialize_rag_system,
    retrieve_documents,
)


class DiscoverChromaBackendsTests(unittest.TestCase):
    @patch("rag_client.chromadb.PersistentClient")
    def test_discovers_collections_and_ignores_non_databases(
        self,
        mocked_persistent_client_class,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            chroma_directory = project_root / "chroma_db"
            empty_directory = project_root / "chroma_empty"
            uninitialized_directory = (
                project_root / "chroma_uninitialized"
            )
            chroma_directory.mkdir()
            empty_directory.mkdir()
            uninitialized_directory.mkdir()
            (chroma_directory / "chroma.sqlite3").touch()
            (empty_directory / "chroma.sqlite3").touch()
            (project_root / "chroma_notes.txt").touch()

            collection = MagicMock()
            collection.name = "nasa_collection"
            collection.count.return_value = 2766

            populated_client = MagicMock()
            populated_client.list_collections.return_value = [
                collection
            ]
            empty_client = MagicMock()
            empty_client.list_collections.return_value = []

            clients_by_path = {
                str(chroma_directory): populated_client,
                str(empty_directory): empty_client,
            }
            mocked_persistent_client_class.side_effect = (
                lambda path: clients_by_path[path]
            )

            with patch("rag_client.Path") as mocked_path_class:
                mocked_path_class.return_value.resolve.return_value.parent = (
                    project_root
                )
                result = discover_chroma_backends()

        self.assertEqual(
            result,
            {
                "chroma_db:nasa_collection": {
                    "directory": str(chroma_directory),
                    "collection_name": "nasa_collection",
                    "display_name": (
                        "nasa_collection "
                        "(chroma_db, 2766 chunks)"
                    ),
                }
            },
        )
        self.assertEqual(
            {
                call.kwargs["path"]
                for call in mocked_persistent_client_class.call_args_list
            },
            {
                str(chroma_directory),
                str(empty_directory),
            },
        )

    @patch("rag_client.chromadb.PersistentClient")
    def test_skips_unreadable_directory_and_collection(
        self,
        mocked_persistent_client_class,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            broken_directory = project_root / "chroma_broken"
            mixed_directory = project_root / "chroma_mixed"
            broken_directory.mkdir()
            mixed_directory.mkdir()
            (broken_directory / "chroma.sqlite3").touch()
            (mixed_directory / "chroma.sqlite3").touch()

            valid_collection = MagicMock()
            valid_collection.name = "valid_collection"
            valid_collection.count.return_value = 12
            broken_collection = MagicMock()
            broken_collection.name = "broken_collection"
            broken_collection.count.side_effect = RuntimeError(
                "damaged collection"
            )

            mixed_client = MagicMock()
            mixed_client.list_collections.return_value = [
                broken_collection,
                valid_collection,
            ]

            def open_client(path):
                if path == str(broken_directory):
                    raise RuntimeError("not a Chroma database")
                return mixed_client

            mocked_persistent_client_class.side_effect = open_client

            with patch("rag_client.Path") as mocked_path_class:
                mocked_path_class.return_value.resolve.return_value.parent = (
                    project_root
                )
                result = discover_chroma_backends()

        self.assertEqual(
            list(result),
            ["chroma_mixed:valid_collection"],
        )
        self.assertEqual(
            result["chroma_mixed:valid_collection"][
                "collection_name"
            ],
            "valid_collection",
        )

    @patch("rag_client.chromadb.PersistentClient")
    def test_returns_empty_mapping_when_no_chroma_paths_exist(
        self,
        mocked_persistent_client_class,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)

            with patch("rag_client.Path") as mocked_path_class:
                mocked_path_class.return_value.resolve.return_value.parent = (
                    project_root
                )
                result = discover_chroma_backends()

        self.assertEqual(result, {})
        mocked_persistent_client_class.assert_not_called()


class InitializeRagSystemTests(unittest.TestCase):
    @patch("rag_client.chromadb.PersistentClient")
    @patch("rag_client.OpenAIEmbeddingFunction")
    def test_opens_existing_collection_with_embedding_config(
        self,
        mocked_embedding_function_class,
        mocked_persistent_client_class,
    ):
        mocked_client = mocked_persistent_client_class.return_value
        mocked_collection = MagicMock()
        mocked_client.get_collection.return_value = (
            mocked_collection
        )

        result = initialize_rag_system(
            chroma_dir="chroma_db_openai",
            collection_name="nasa_space_missions_text",
            openai_api_key="test-key",
            openai_base_url="https://example.test/v1",
            embedding_model="test-embedding-model",
        )

        mocked_embedding_function_class.assert_called_once_with(
            api_key="test-key",
            api_base="https://example.test/v1",
            model_name="test-embedding-model",
        )
        mocked_persistent_client_class.assert_called_once_with(
            path="chroma_db_openai"
        )
        mocked_client.get_collection.assert_called_once_with(
            name="nasa_space_missions_text",
            embedding_function=(
                mocked_embedding_function_class.return_value
            ),
        )
        self.assertEqual(
            result,
            (mocked_collection, True, None),
        )

    @patch.dict("rag_client.os.environ", {}, clear=True)
    @patch("rag_client.chromadb.PersistentClient")
    def test_rejects_missing_api_key(
        self,
        mocked_persistent_client_class,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "OpenAI API key must not be empty",
        ):
            initialize_rag_system(
                chroma_dir="chroma_db_openai",
                collection_name="nasa_space_missions_text",
            )

        mocked_persistent_client_class.assert_not_called()


class RetrieveDocumentsTests(unittest.TestCase):
    def test_queries_with_runtime_top_k_and_no_filter(self):
        collection = MagicMock()
        expected_result = {
            "documents": [["Apollo 11 context"]],
            "metadatas": [[{"mission": "apollo11"}]],
            "distances": [[0.12]],
        }
        collection.query.return_value = expected_result

        result = retrieve_documents(
            collection=collection,
            query="  What happened during Apollo 11?  ",
            n_results=5,
            mission_filter="all",
        )

        collection.query.assert_called_once_with(
            query_texts=["What happened during Apollo 11?"],
            n_results=5,
            where=None,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )
        self.assertIs(result, expected_result)

    def test_applies_canonical_mission_filter(self):
        collection = MagicMock()

        retrieve_documents(
            collection=collection,
            query="What caused the launch failure?",
            n_results=3,
            mission_filter="STS-51L",
        )

        collection.query.assert_called_once_with(
            query_texts=["What caused the launch failure?"],
            n_results=3,
            where={
                "mission": {
                    "$eq": "challenger",
                }
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    def test_rejects_invalid_retrieval_inputs(self):
        collection = MagicMock()

        invalid_cases = [
            {
                "query": "   ",
                "n_results": 3,
                "mission_filter": None,
                "message": "query must not be empty",
            },
            {
                "query": "Apollo 13",
                "n_results": 0,
                "mission_filter": None,
                "message": "n_results must be a positive integer",
            },
            {
                "query": "Gemini",
                "n_results": 3,
                "mission_filter": "Gemini",
                "message": "Unsupported mission filter",
            },
        ]

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    ValueError,
                    case["message"],
                ):
                    retrieve_documents(
                        collection=collection,
                        query=case["query"],
                        n_results=case["n_results"],
                        mission_filter=case["mission_filter"],
                    )

        collection.query.assert_not_called()


class FormatContextTests(unittest.TestCase):
    def test_formats_cited_blocks_with_provenance(self):
        context = format_context(
            documents=[
                "Apollo report evidence.",
                "Challenger transcript evidence.",
            ],
            metadatas=[
                {
                    "mission": "apollo11",
                    "source_file": "apollo_report.txt",
                    "filepath": "data_text/apollo11/apollo_report.txt",
                    "source_type": "report",
                    "page_start": 13,
                    "page_end": 14,
                },
                {
                    "mission": "challenger",
                    "source_file": "challenger_transcript.txt",
                    "source_path": (
                        "data_text/challenger/"
                        "challenger_transcript.txt"
                    ),
                    "source_type": "transcript",
                    "source_line_start": 42,
                    "source_line_end": 43,
                },
            ],
        )

        self.assertTrue(
            context.startswith("RETRIEVED DOCUMENTS")
        )
        self.assertIn("[DOCUMENT 1]", context)
        self.assertIn("MISSION = Apollo 11", context)
        self.assertIn("SOURCE = apollo_report.txt", context)
        self.assertIn("PAGES = 13-14", context)
        self.assertIn("[DOCUMENT 2]", context)
        self.assertIn("MISSION = Challenger", context)
        self.assertIn("LINES = 42-43", context)
        self.assertIn("\n\n---\n\n", context)

    def test_deduplicates_chunks_while_preserving_order(self):
        context = format_context(
            documents=[
                "First retrieved chunk.",
                "  first   retrieved CHUNK.  ",
                "Second retrieved chunk.",
            ],
            metadatas=[
                {
                    "mission": "apollo11",
                    "source": "source-one",
                },
                {
                    "mission": "apollo11",
                    "source": "duplicate-source",
                },
                {
                    "mission": "apollo13",
                    "source": "source-two",
                },
            ],
        )

        self.assertEqual(context.count("[DOCUMENT "), 2)
        self.assertNotIn("duplicate-source", context)
        self.assertLess(
            context.index("First retrieved chunk."),
            context.index("Second retrieved chunk."),
        )

    def test_rejects_misaligned_or_malformed_inputs(self):
        with self.assertRaisesRegex(
            ValueError,
            "must have equal lengths",
        ):
            format_context(
                documents=["one document"],
                metadatas=[],
            )

        with self.assertRaisesRegex(
            ValueError,
            "metadata value must be a dictionary",
        ):
            format_context(
                documents=["one document"],
                metadatas=[None],
            )


if __name__ == "__main__":
    unittest.main()
