import asyncio
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ragas_evaluator


class RecordingScorer:
    def __init__(self, value):
        self.value = value
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        return SimpleNamespace(value=self.value)


class ScoreMetricsAsyncTests(unittest.TestCase):
    def test_scores_each_metric_and_preserves_names(self):
        relevancy = RecordingScorer(0.8)
        faithfulness = RecordingScorer(0.9)
        scorers = {
            "response_relevancy": relevancy,
            "faithfulness": faithfulness,
        }
        metric_inputs = {
            "response_relevancy": {
                "user_input": "Question",
                "response": "Answer",
            },
            "faithfulness": {
                "user_input": "Question",
                "response": "Answer",
                "retrieved_contexts": ["Context"],
            },
        }

        scores = asyncio.run(
            ragas_evaluator._score_metrics_async(
                scorers=scorers,
                metric_inputs=metric_inputs,
            )
        )

        self.assertEqual(
            scores,
            {
                "response_relevancy": 0.8,
                "faithfulness": 0.9,
            },
        )
        self.assertEqual(
            relevancy.calls,
            [metric_inputs["response_relevancy"]],
        )
        self.assertEqual(
            faithfulness.calls,
            [metric_inputs["faithfulness"]],
        )


class EvaluateResponseQualityTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)

        self.openai_class = self.patches.enter_context(
            patch(
                "ragas_evaluator.OpenAI",
                create=True,
            )
        )
        self.async_openai_class = self.patches.enter_context(
            patch(
                "ragas_evaluator.AsyncOpenAI",
                create=True,
            )
        )
        self.llm_factory = self.patches.enter_context(
            patch("ragas_evaluator.llm_factory")
        )
        self.embeddings_class = self.patches.enter_context(
            patch("ragas_evaluator.OpenAIEmbeddings")
        )
        self.answer_relevancy_class = self.patches.enter_context(
            patch("ragas_evaluator.AnswerRelevancy")
        )
        self.faithfulness_class = self.patches.enter_context(
            patch("ragas_evaluator.Faithfulness")
        )
        self.context_precision_class = self.patches.enter_context(
            patch("ragas_evaluator.ContextPrecision")
        )
        self.context_recall_class = self.patches.enter_context(
            patch("ragas_evaluator.ContextRecall")
        )
        self.factual_correctness_class = self.patches.enter_context(
            patch("ragas_evaluator.FactualCorrectness")
        )
        self.score_metrics_async = self.patches.enter_context(
            patch(
                "ragas_evaluator._score_metrics_async",
                new_callable=AsyncMock,
            )
        )

        self.async_client = object()
        self.evaluator_llm = SimpleNamespace(
            model_args={
                "temperature": 0.01,
                "top_p": 0.1,
                "max_tokens": 4096,
            }
        )
        self.evaluator_embeddings = object()
        self.async_openai_class.return_value = (
            self.async_client
        )
        self.llm_factory.return_value = self.evaluator_llm
        self.embeddings_class.return_value = self.evaluator_embeddings

    def evaluate(self, **changes):
        arguments = {
            "question": "  What happened?  ",
            "answer": "  A grounded answer.  ",
            "contexts": ["  Retrieved context.  "],
            "evaluator_model": "  judge-model  ",
            "embedding_model": "  embedding-model  ",
            "openai_api_key": "  test-key  ",
            "openai_base_url": "  https://example.test/v1  ",
        }
        arguments.update(changes)
        return ragas_evaluator.evaluate_response_quality(**arguments)

    def test_rejects_malformed_inputs_before_setup(self):
        invalid_cases = [
            (
                {"question": "   "},
                "question must be a non-empty string",
            ),
            (
                {"answer": None},
                "answer must be a non-empty string",
            ),
            (
                {"contexts": []},
                "contexts must be a non-empty list",
            ),
            (
                {"contexts": "not a list"},
                "contexts must be a non-empty list",
            ),
            (
                {"contexts": ["valid", "   "]},
                "each context must be a non-empty string",
            ),
            (
                {"reference": "   "},
                "reference must be a non-empty string when provided",
            ),
            (
                {"evaluator_model": "   "},
                "evaluator_model must not be empty",
            ),
            (
                {"embedding_model": "   "},
                "embedding_model must not be empty",
            ),
            (
                {"openai_api_key": "   "},
                "OpenAI API key must not be empty",
            ),
        ]

        for changes, expected_error in invalid_cases:
            with self.subTest(changes=changes):
                self.assertEqual(
                    self.evaluate(**changes),
                    {"error": expected_error},
                )

        self.openai_class.assert_not_called()
        self.async_openai_class.assert_not_called()
        self.score_metrics_async.assert_not_awaited()

    def test_reports_when_ragas_is_unavailable(self):
        with patch.object(
            ragas_evaluator,
            "RAGAS_AVAILABLE",
            False,
        ):
            result = self.evaluate()

        self.assertEqual(
            result,
            {"error": "RAGAS is not available"},
        )
        self.openai_class.assert_not_called()
        self.async_openai_class.assert_not_called()

    def test_evaluates_required_metrics_without_reference(self):
        self.score_metrics_async.return_value = {
            "response_relevancy": 0.8,
            "faithfulness": 0.9,
        }

        result = self.evaluate()

        self.assertEqual(
            result,
            {
                "response_relevancy": 0.8,
                "faithfulness": 0.9,
            },
        )
        self.openai_class.assert_not_called()
        self.async_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.test/v1",
        )
        self.llm_factory.assert_called_once_with(
            model="judge-model",
            provider="openai",
            client=self.async_client,
            max_tokens=4096,
        )
        self.embeddings_class.assert_called_once_with(
            client=self.async_client,
            model="embedding-model",
        )
        self.answer_relevancy_class.assert_called_once_with(
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
        )
        self.faithfulness_class.assert_called_once_with(
            llm=self.evaluator_llm,
        )
        self.context_precision_class.assert_not_called()
        self.context_recall_class.assert_not_called()
        self.factual_correctness_class.assert_not_called()

        call = self.score_metrics_async.await_args.kwargs
        self.assertEqual(
            call["metric_inputs"],
            {
                "response_relevancy": {
                    "user_input": "What happened?",
                    "response": "A grounded answer.",
                },
                "faithfulness": {
                    "user_input": "What happened?",
                    "response": "A grounded answer.",
                    "retrieved_contexts": ["Retrieved context."],
                },
            },
        )

    def test_default_gpt54_evaluator_uses_compatible_arguments(self):
        self.score_metrics_async.return_value = {
            "response_relevancy": 0.8,
            "faithfulness": 0.9,
        }

        result = ragas_evaluator.evaluate_response_quality(
            question="Question",
            answer="Answer",
            contexts=["Context"],
            embedding_model="embedding-model",
            openai_api_key="test-key",
            openai_base_url="https://example.test/v1",
        )

        self.assertEqual(
            result,
            {
                "response_relevancy": 0.8,
                "faithfulness": 0.9,
            },
        )
        self.llm_factory.assert_called_once_with(
            model="gpt-5.4-mini",
            provider="openai",
            client=self.async_client,
            max_tokens=4096,
            reasoning_effort="low",
        )
        self.assertEqual(
            self.evaluator_llm.model_args,
            {
                "temperature": 1.0,
                "max_completion_tokens": 4096,
                "reasoning_effort": "low",
            },
        )

    def test_uses_minimal_reasoning_for_legacy_gpt5_nano(self):
        self.score_metrics_async.return_value = {
            "response_relevancy": 0.8,
            "faithfulness": 0.9,
        }

        result = self.evaluate(
            evaluator_model="  gpt-5-nano  "
        )

        self.assertEqual(
            result,
            {
                "response_relevancy": 0.8,
                "faithfulness": 0.9,
            },
        )
        self.llm_factory.assert_called_once_with(
            model="gpt-5-nano",
            provider="openai",
            client=self.async_client,
            max_tokens=4096,
            reasoning_effort="minimal",
        )

    def test_evaluates_reference_metrics_and_retrieval_f1(self):
        self.score_metrics_async.return_value = {
            "response_relevancy": 0.8,
            "faithfulness": 0.9,
            "context_precision": 0.8,
            "context_recall": 0.5,
            "factual_correctness": 0.7,
        }

        result = self.evaluate(
            reference="  Expected answer.  ",
        )

        self.assertEqual(
            set(result),
            {
                "response_relevancy",
                "faithfulness",
                "context_precision",
                "context_recall",
                "factual_correctness",
                "retrieval_f1",
            },
        )
        self.assertAlmostEqual(
            result["retrieval_f1"],
            2 * 0.8 * 0.5 / (0.8 + 0.5),
        )
        self.context_precision_class.assert_called_once_with(
            llm=self.evaluator_llm,
        )
        self.context_recall_class.assert_called_once_with(
            llm=self.evaluator_llm,
        )
        self.factual_correctness_class.assert_called_once_with(
            llm=self.evaluator_llm,
            mode="f1",
        )

        call = self.score_metrics_async.await_args.kwargs
        self.assertEqual(
            call["metric_inputs"]["context_precision"],
            {
                "user_input": "What happened?",
                "reference": "Expected answer.",
                "retrieved_contexts": ["Retrieved context."],
            },
        )
        self.assertEqual(
            call["metric_inputs"]["factual_correctness"],
            {
                "response": "A grounded answer.",
                "reference": "Expected answer.",
            },
        )

    def test_retrieval_f1_is_zero_when_precision_and_recall_are_zero(self):
        self.score_metrics_async.return_value = {
            "response_relevancy": 0.8,
            "faithfulness": 0.9,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "factual_correctness": 0.0,
        }

        result = self.evaluate(reference="Expected answer.")

        self.assertEqual(result["retrieval_f1"], 0.0)

    def test_returns_structured_error_when_scoring_fails(self):
        self.score_metrics_async.side_effect = RuntimeError(
            "judge unavailable"
        )

        result = self.evaluate()

        self.assertEqual(
            result,
            {
                "error": (
                    "Evaluation failed: "
                    "RuntimeError: judge unavailable"
                )
            },
        )

    def test_returns_structured_error_when_setup_fails(self):
        self.async_openai_class.side_effect = ValueError(
            "invalid base URL"
        )

        result = self.evaluate()

        self.assertEqual(
            result,
            {
                "error": (
                    "Evaluation setup failed: "
                    "ValueError: invalid base URL"
                )
            },
        )


if __name__ == "__main__":
    unittest.main()
