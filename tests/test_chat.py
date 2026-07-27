import unittest
from unittest.mock import patch

from chat import (
    evaluate_response_quality,
    generate_response,
    initialize_rag_system,
    retrieve_documents,
)


class EvaluateResponseQualityWrapperTests(unittest.TestCase):
    @patch.dict(
        "chat.os.environ",
        {"OPENAI_BASE_URL": "https://example.test/v1"},
        clear=True,
    )
    @patch(
        "chat.ragas_evaluator.evaluate_response_quality"
    )
    def test_forwards_sidebar_openai_configuration(
        self,
        mocked_evaluate_response_quality,
    ):
        expected_scores = {
            "response_relevancy": 0.9,
            "faithfulness": 0.8,
        }
        mocked_evaluate_response_quality.return_value = (
            expected_scores
        )

        result = evaluate_response_quality(
            question="What happened?",
            answer="A grounded answer.",
            contexts=["Retrieved evidence."],
            openai_key="sidebar-key",
        )

        self.assertEqual(result, expected_scores)
        mocked_evaluate_response_quality.assert_called_once_with(
            question="What happened?",
            answer="A grounded answer.",
            contexts=["Retrieved evidence."],
            openai_api_key="sidebar-key",
            openai_base_url="https://example.test/v1",
        )


class InitializeRagSystemWrapperTests(unittest.TestCase):
    @patch.dict(
        "chat.os.environ",
        {"OPENAI_BASE_URL": "https://example.test/v1"},
        clear=True,
    )
    @patch("chat.rag_client.initialize_rag_system")
    def test_forwards_sidebar_openai_configuration(
        self,
        mocked_initialize_rag_system,
    ):
        expected_result = (object(), True, None)
        mocked_initialize_rag_system.return_value = expected_result

        result = initialize_rag_system(
            chroma_dir="chroma_db",
            collection_name="nasa_collection",
            openai_key="sidebar-key",
        )

        self.assertEqual(result, expected_result)
        mocked_initialize_rag_system.assert_called_once_with(
            chroma_dir="chroma_db",
            collection_name="nasa_collection",
            openai_api_key="sidebar-key",
            openai_base_url="https://example.test/v1",
        )


class RetrieveDocumentsWrapperTests(unittest.TestCase):
    @patch("chat.rag_client.retrieve_documents")
    def test_forwards_runtime_top_k_and_mission_filter(
        self,
        mocked_retrieve_documents,
    ):
        collection = object()
        expected_result = {
            "documents": [["Apollo 13 evidence."]],
            "metadatas": [[{"mission": "apollo13"}]],
            "distances": [[0.1]],
        }
        mocked_retrieve_documents.return_value = expected_result

        result = retrieve_documents(
            collection=collection,
            query="What happened during Apollo 13?",
            n_results=7,
            mission_filter="Apollo 13",
        )

        self.assertIs(result, expected_result)
        mocked_retrieve_documents.assert_called_once_with(
            collection,
            "What happened during Apollo 13?",
            7,
            "Apollo 13",
        )


class GenerateResponseWrapperTests(unittest.TestCase):
    @patch.dict(
        "chat.os.environ",
        {"OPENAI_BASE_URL": "https://example.test/v1"},
        clear=True,
    )
    @patch("chat.llm_client.generate_response")
    def test_forwards_runtime_configuration(
        self,
        mocked_generate_response,
    ):
        mocked_generate_response.return_value = (
            "Grounded answer [DOCUMENT 1]."
        )
        history = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]

        answer = generate_response(
            openai_key="test-key",
            user_message="What happened?",
            context="[DOCUMENT 1]\nRetrieved evidence.",
            conversation_history=history,
            model="test-chat-model",
        )

        self.assertEqual(
            answer,
            "Grounded answer [DOCUMENT 1].",
        )
        mocked_generate_response.assert_called_once_with(
            openai_key="test-key",
            user_message="What happened?",
            context="[DOCUMENT 1]\nRetrieved evidence.",
            conversation_history=history,
            model="test-chat-model",
            openai_base_url="https://example.test/v1",
        )

    @patch("chat.llm_client.generate_response")
    def test_propagates_generation_errors(
        self,
        mocked_generate_response,
    ):
        mocked_generate_response.side_effect = RuntimeError(
            "API unavailable"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "API unavailable",
        ):
            generate_response(
                openai_key="test-key",
                user_message="What happened?",
                context="[DOCUMENT 1]\nRetrieved evidence.",
                conversation_history=[],
                model="test-chat-model",
            )


if __name__ == "__main__":
    unittest.main()
