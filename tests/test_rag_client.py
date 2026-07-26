import unittest
from unittest.mock import MagicMock, patch

from rag_client import initialize_rag_system


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


if __name__ == "__main__":
    unittest.main()
