import unittest
from unittest.mock import MagicMock, patch

from llm_client import DEVELOPER_PROMPT, generate_response


class GenerateResponseTests(unittest.TestCase):
    @staticmethod
    def configure_response(
        mocked_openai_class,
        content="  Grounded answer [DOCUMENT 1].  ",
    ):
        mocked_response = MagicMock()
        mocked_response.choices = [MagicMock()]
        mocked_response.choices[0].message.content = content
        mocked_openai_class.return_value.chat.completions.create.return_value = (
            mocked_response
        )

    @patch("llm_client.OpenAI")
    def test_builds_grounded_messages_and_returns_answer(
        self,
        mocked_openai_class,
    ):
        self.configure_response(mocked_openai_class)
        history = [
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one"},
            {"role": "user", "content": "Question two"},
            {"role": "assistant", "content": "Answer two"},
            {"role": "user", "content": "Question three"},
            {"role": "assistant", "content": "Answer three"},
        ]
        context = (
            "RETRIEVED DOCUMENTS\n"
            "[DOCUMENT 1]\n"
            "Apollo 11 landed on the Moon."
        )

        answer = generate_response(
            openai_key="test-key",
            user_message="When did it land?",
            context=context,
            conversation_history=history,
            model="test-chat-model",
            openai_base_url="https://example.test/v1",
            max_history_messages=4,
        )

        self.assertEqual(answer, "Grounded answer [DOCUMENT 1].")
        mocked_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.test/v1",
        )

        create_call = (
            mocked_openai_class.return_value
            .chat.completions.create
        )
        create_call.assert_called_once()
        request = create_call.call_args.kwargs
        self.assertEqual(request["model"], "test-chat-model")

        messages = request["messages"]
        self.assertEqual(
            messages[0],
            {
                "role": "developer",
                "content": DEVELOPER_PROMPT,
            },
        )
        self.assertEqual(messages[1:-1], history[-4:])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn(
            "USER QUESTION:\nWhen did it land?",
            messages[-1]["content"],
        )
        self.assertIn(
            "RETRIEVED DOCUMENTS:\n",
            messages[-1]["content"],
        )
        self.assertIn(context, messages[-1]["content"])

        self.assertIn("NASA mission expert", DEVELOPER_PROMPT)
        self.assertIn("[DOCUMENT N]", DEVELOPER_PROMPT)
        self.assertIn("quoted source material", DEVELOPER_PROMPT)
        self.assertIn("insufficient", DEVELOPER_PROMPT)

    @patch.dict(
        "llm_client.os.environ",
        {"OPENAI_BASE_URL": "https://environment.test/v1"},
        clear=True,
    )
    @patch("llm_client.OpenAI")
    def test_uses_environment_url_and_handles_empty_context(
        self,
        mocked_openai_class,
    ):
        self.configure_response(
            mocked_openai_class,
            content="The retrieved context is insufficient.",
        )

        answer = generate_response(
            openai_key="test-key",
            user_message="What happened?",
            context="   ",
            conversation_history=[
                {"role": "user", "content": "Old question"},
                {"role": "assistant", "content": "Old answer"},
            ],
            model="test-chat-model",
            max_history_messages=0,
        )

        self.assertEqual(
            answer,
            "The retrieved context is insufficient.",
        )
        mocked_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://environment.test/v1",
        )
        messages = (
            mocked_openai_class.return_value
            .chat.completions.create.call_args.kwargs["messages"]
        )
        self.assertEqual(len(messages), 2)
        self.assertIn(
            "No retrieved documents were provided.",
            messages[-1]["content"],
        )

    @patch("llm_client.OpenAI")
    def test_rejects_invalid_function_inputs(
        self,
        mocked_openai_class,
    ):
        valid_arguments = {
            "openai_key": "test-key",
            "user_message": "What happened?",
            "context": "Retrieved evidence",
            "conversation_history": [],
            "model": "test-chat-model",
        }
        invalid_cases = [
            (
                {"openai_key": "   "},
                "openai_key must not be empty",
            ),
            (
                {"user_message": "   "},
                "user_message must not be empty",
            ),
            (
                {"context": None},
                "context must be a string",
            ),
            (
                {"conversation_history": "not a list"},
                "conversation_history must be a list",
            ),
            (
                {"model": "   "},
                "model must not be empty",
            ),
            (
                {"max_history_messages": -1},
                "max_history_messages must be a non-negative integer",
            ),
            (
                {"max_history_messages": True},
                "max_history_messages must be a non-negative integer",
            ),
        ]

        for changes, expected_message in invalid_cases:
            arguments = valid_arguments | changes
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(
                    ValueError,
                    expected_message,
                ):
                    generate_response(**arguments)

        mocked_openai_class.assert_not_called()

    @patch("llm_client.OpenAI")
    def test_rejects_malformed_history_turns(
        self,
        mocked_openai_class,
    ):
        invalid_histories = [
            ["not a dictionary"],
            [{"role": "developer", "content": "Not user history"}],
            [{"role": "user", "content": "   "}],
        ]

        for history in invalid_histories:
            with self.subTest(history=history):
                with self.assertRaises(ValueError):
                    generate_response(
                        openai_key="test-key",
                        user_message="What happened?",
                        context="Retrieved evidence",
                        conversation_history=history,
                        model="test-chat-model",
                    )

        mocked_openai_class.assert_not_called()

    @patch("llm_client.OpenAI")
    def test_raises_when_api_returns_no_choices(
        self,
        mocked_openai_class,
    ):
        mocked_response = MagicMock()
        mocked_response.choices = []
        mocked_openai_class.return_value.chat.completions.create.return_value = (
            mocked_response
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "returned no choices",
        ):
            generate_response(
                openai_key="test-key",
                user_message="What happened?",
                context="Retrieved evidence",
                conversation_history=[],
                model="test-chat-model",
            )

    @patch("llm_client.OpenAI")
    def test_raises_when_api_returns_empty_answer(
        self,
        mocked_openai_class,
    ):
        self.configure_response(
            mocked_openai_class,
            content="   ",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "returned an empty answer",
        ):
            generate_response(
                openai_key="test-key",
                user_message="What happened?",
                context="Retrieved evidence",
                conversation_history=[],
                model="test-chat-model",
            )


if __name__ == "__main__":
    unittest.main()
