import unittest
from unittest.mock import patch

from chat import generate_response


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
