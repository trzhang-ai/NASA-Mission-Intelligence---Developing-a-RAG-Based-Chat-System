import unittest
from unittest.mock import MagicMock, patch

from rag_client import (
    initialize_rag_system,
    retrieve_documents,
)


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


if __name__ == "__main__":
    unittest.main()
