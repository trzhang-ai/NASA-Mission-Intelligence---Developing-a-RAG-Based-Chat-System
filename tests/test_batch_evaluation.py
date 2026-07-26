import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from batch_evaluation import (
    EVALUATION_METRICS,
    REQUIRED_QUESTION_FIELDS,
    build_argument_parser,
    evaluate_batch,
    evaluate_one_question,
    generate_answer_for_question,
    load_test_questions,
    main,
    print_batch_summary,
    retrieve_question_evidence,
    run_batch_evaluation,
    summarize_batch_results,
    write_batch_report,
)


class LoadTestQuestionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.dataset_path = (
            Path(self.temporary_directory.name) / "questions.json"
        )

    def write_json(self, payload):
        self.dataset_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    @staticmethod
    def valid_question():
        return {
            "id": "q01",
            "mission": "apollo11",
            "category": "overview",
            "user_input": "What was the mission objective?",
            "reference": "Land astronauts and return safely.",
        }

    def test_loads_project_dataset(self):
        project_root = Path(__file__).resolve().parents[1]

        questions = load_test_questions(
            project_root / "test_questions.json"
        )

        self.assertEqual(len(questions), 17)
        self.assertEqual(
            {question["mission"] for question in questions},
            {"apollo11", "apollo13", "challenger"},
        )
        self.assertGreaterEqual(
            len(
                {
                    question["category"]
                    for question in questions
                }
            ),
            5,
        )
        self.assertTrue(
            all(
                REQUIRED_QUESTION_FIELDS <= question.keys()
                for question in questions
            )
        )

    def test_rejects_missing_file(self):
        with self.assertRaisesRegex(
            FileNotFoundError,
            "Evaluation dataset does not exist",
        ):
            load_test_questions(self.dataset_path)

    def test_rejects_invalid_json(self):
        self.dataset_path.write_text(
            "{not valid JSON",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Evaluation dataset is not valid JSON",
        ):
            load_test_questions(self.dataset_path)

    def test_rejects_non_object_top_level_value(self):
        self.write_json([self.valid_question()])

        with self.assertRaisesRegex(
            ValueError,
            "Evaluation dataset must be an object",
        ):
            load_test_questions(self.dataset_path)

    def test_rejects_missing_or_empty_questions_list(self):
        invalid_payloads = [
            {},
            {"questions": []},
            {"questions": "not a list"},
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self.write_json(payload)

                with self.assertRaisesRegex(
                    ValueError,
                    "must contain a non-empty questions list",
                ):
                    load_test_questions(self.dataset_path)

    def test_rejects_non_object_question(self):
        self.write_json({"questions": ["not an object"]})

        with self.assertRaisesRegex(
            ValueError,
            "Question at index 0 must be an object",
        ):
            load_test_questions(self.dataset_path)

    def test_rejects_missing_required_fields(self):
        question = self.valid_question()
        del question["reference"]
        self.write_json({"questions": [question]})

        with self.assertRaisesRegex(
            ValueError,
            "missing fields.*reference",
        ):
            load_test_questions(self.dataset_path)

    def test_rejects_blank_required_fields(self):
        question = self.valid_question()
        question["user_input"] = "   "
        self.write_json({"questions": [question]})

        with self.assertRaisesRegex(
            ValueError,
            "field 'user_input' must be a non-empty string",
        ):
                load_test_questions(self.dataset_path)


class RetrieveQuestionEvidenceTests(unittest.TestCase):
    @patch("batch_evaluation.rag_client.format_context")
    @patch("batch_evaluation.rag_client.retrieve_documents")
    def test_retrieves_and_formats_question_evidence(
        self,
        mocked_retrieve_documents,
        mocked_format_context,
    ):
        collection = object()
        question = {
            "user_input": "What happened during Apollo 13?",
            "mission": "apollo13",
        }
        documents = [
            "First retrieved chunk.",
            "Second retrieved chunk.",
        ]
        metadatas = [
            {"mission": "apollo13", "source": "report-a"},
            {"mission": "apollo13", "source": "report-b"},
        ]
        mocked_retrieve_documents.return_value = {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [[0.1, 0.2]],
        }
        mocked_format_context.return_value = (
            "[DOCUMENT 1]\nFirst retrieved chunk."
        )

        result = retrieve_question_evidence(
            collection=collection,
            question=question,
            top_k=2,
        )

        self.assertEqual(
            result,
            (
                documents,
                metadatas,
                "[DOCUMENT 1]\nFirst retrieved chunk.",
            ),
        )
        mocked_retrieve_documents.assert_called_once_with(
            collection=collection,
            query="What happened during Apollo 13?",
            n_results=2,
            mission_filter="apollo13",
        )
        mocked_format_context.assert_called_once_with(
            documents=documents,
            metadatas=metadatas,
        )

    @patch("batch_evaluation.rag_client.retrieve_documents")
    def test_rejects_malformed_retrieval_container(
        self,
        mocked_retrieve_documents,
    ):
        malformed_results = [
            None,
            {},
            {"documents": [], "metadatas": []},
        ]

        for result in malformed_results:
            with self.subTest(result=result):
                mocked_retrieve_documents.return_value = result

                with self.assertRaisesRegex(
                    RuntimeError,
                    "Retrieval returned malformed results",
                ):
                    retrieve_question_evidence(
                        collection=object(),
                        question={
                            "user_input": "Question",
                            "mission": "apollo11",
                        },
                        top_k=3,
                    )

    @patch("batch_evaluation.rag_client.retrieve_documents")
    def test_rejects_non_list_document_or_metadata_groups(
        self,
        mocked_retrieve_documents,
    ):
        malformed_results = [
            (
                {
                    "documents": ["not a nested list"],
                    "metadatas": [[{"source": "report"}]],
                },
                "Retrieved documents must be a list",
            ),
            (
                {
                    "documents": [["Retrieved chunk"]],
                    "metadatas": ["not a nested list"],
                },
                "Retrieved metadatas must be a list",
            ),
        ]

        for result, expected_error in malformed_results:
            with self.subTest(result=result):
                mocked_retrieve_documents.return_value = result

                with self.assertRaisesRegex(
                    RuntimeError,
                    expected_error,
                ):
                    retrieve_question_evidence(
                        collection=object(),
                        question={
                            "user_input": "Question",
                            "mission": "apollo11",
                        },
                        top_k=3,
                    )

    @patch("batch_evaluation.rag_client.retrieve_documents")
    def test_rejects_empty_document_results(
        self,
        mocked_retrieve_documents,
    ):
        mocked_retrieve_documents.return_value = {
            "documents": [[]],
            "metadatas": [[]],
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "Retrieval returned no documents",
        ):
            retrieve_question_evidence(
                collection=object(),
                question={
                    "user_input": "Question",
                    "mission": "apollo11",
                },
                top_k=3,
            )


class GenerateAnswerForQuestionTests(unittest.TestCase):
    @patch("batch_evaluation.llm_client.generate_response")
    @patch("batch_evaluation.retrieve_question_evidence")
    def test_generates_grounded_answer_from_formatted_context(
        self,
        mocked_retrieve_question_evidence,
        mocked_generate_response,
    ):
        collection = object()
        question = {
            "user_input": "What happened during Apollo 13?",
            "mission": "apollo13",
        }
        documents = ["Retrieved chunk."]
        metadatas = [
            {"mission": "apollo13", "source": "report"}
        ]
        formatted_context = (
            "[DOCUMENT 1]\nRetrieved chunk."
        )
        mocked_retrieve_question_evidence.return_value = (
            documents,
            metadatas,
            formatted_context,
        )
        mocked_generate_response.return_value = (
            "Grounded answer [DOCUMENT 1]."
        )

        result = generate_answer_for_question(
            collection=collection,
            question=question,
            top_k=4,
            openai_api_key="test-key",
            generator_model="test-generator",
            openai_base_url="https://example.test/v1",
        )

        self.assertEqual(
            result,
            (
                documents,
                metadatas,
                "Grounded answer [DOCUMENT 1].",
            ),
        )
        mocked_retrieve_question_evidence.assert_called_once_with(
            collection=collection,
            question=question,
            top_k=4,
        )
        mocked_generate_response.assert_called_once_with(
            openai_key="test-key",
            user_message="What happened during Apollo 13?",
            context=formatted_context,
            conversation_history=[],
            model="test-generator",
            openai_base_url="https://example.test/v1",
        )

    @patch("batch_evaluation.llm_client.generate_response")
    @patch("batch_evaluation.retrieve_question_evidence")
    def test_retrieval_error_stops_generation(
        self,
        mocked_retrieve_question_evidence,
        mocked_generate_response,
    ):
        mocked_retrieve_question_evidence.side_effect = RuntimeError(
            "Retrieval failed"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Retrieval failed",
        ):
            generate_answer_for_question(
                collection=object(),
                question={
                    "user_input": "Question",
                    "mission": "apollo11",
                },
                top_k=3,
                openai_api_key="test-key",
                generator_model="test-generator",
            )

        mocked_generate_response.assert_not_called()

    @patch("batch_evaluation.llm_client.generate_response")
    @patch("batch_evaluation.retrieve_question_evidence")
    def test_generation_error_is_propagated(
        self,
        mocked_retrieve_question_evidence,
        mocked_generate_response,
    ):
        mocked_retrieve_question_evidence.return_value = (
            ["Retrieved chunk."],
            [{"mission": "apollo11"}],
            "[DOCUMENT 1]\nRetrieved chunk.",
        )
        mocked_generate_response.side_effect = RuntimeError(
            "Generation failed"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Generation failed",
        ):
            generate_answer_for_question(
                collection=object(),
                question={
                    "user_input": "Question",
                    "mission": "apollo11",
                },
                top_k=3,
                openai_api_key="test-key",
                generator_model="test-generator",
            )


class EvaluateOneQuestionTests(unittest.TestCase):
    @patch(
        "batch_evaluation.ragas_evaluator.evaluate_response_quality"
    )
    @patch("batch_evaluation.generate_answer_for_question")
    def test_returns_complete_per_question_result(
        self,
        mocked_generate_answer,
        mocked_evaluate_quality,
    ):
        collection = object()
        question = {
            "id": "q01",
            "mission": "apollo11",
            "category": "overview",
            "user_input": "What was the mission objective?",
            "reference": "Land and return safely.",
            "source": "Apollo 11 report",
        }
        documents = ["Retrieved mission evidence."]
        metadatas = [
            {"mission": "apollo11", "source": "report"}
        ]
        mocked_generate_answer.return_value = (
            documents,
            metadatas,
            "Grounded answer [DOCUMENT 1].",
        )
        mocked_evaluate_quality.return_value = {
            "response_relevancy": 0.8,
            "faithfulness": 0.9,
            "context_precision": 0.7,
            "context_recall": 0.6,
            "factual_correctness": 0.75,
            "retrieval_f1": 0.646,
        }

        result = evaluate_one_question(
            collection=collection,
            question=question,
            top_k=5,
            openai_api_key="test-key",
            generator_model="test-generator",
            evaluator_model="test-evaluator",
            embedding_model="test-embedding",
            openai_base_url="https://example.test/v1",
        )

        self.assertEqual(result["id"], "q01")
        self.assertEqual(
            result["response"],
            "Grounded answer [DOCUMENT 1].",
        )
        self.assertEqual(
            result["retrieved_contexts"],
            documents,
        )
        self.assertEqual(
            result["retrieved_metadatas"],
            metadatas,
        )
        self.assertEqual(result["context_count"], 1)
        self.assertEqual(result["faithfulness"], 0.9)
        self.assertEqual(result["retrieval_f1"], 0.646)

        mocked_generate_answer.assert_called_once_with(
            collection=collection,
            question=question,
            top_k=5,
            openai_api_key="test-key",
            generator_model="test-generator",
            openai_base_url="https://example.test/v1",
        )
        mocked_evaluate_quality.assert_called_once_with(
            question="What was the mission objective?",
            answer="Grounded answer [DOCUMENT 1].",
            contexts=documents,
            evaluator_model="test-evaluator",
            reference="Land and return safely.",
            embedding_model="test-embedding",
            openai_api_key="test-key",
            openai_base_url="https://example.test/v1",
        )

    @patch(
        "batch_evaluation.ragas_evaluator.evaluate_response_quality"
    )
    @patch("batch_evaluation.generate_answer_for_question")
    def test_preserves_structured_evaluator_error(
        self,
        mocked_generate_answer,
        mocked_evaluate_quality,
    ):
        mocked_generate_answer.return_value = (
            ["Retrieved evidence."],
            [{"source": "report"}],
            "Generated answer.",
        )
        mocked_evaluate_quality.return_value = {
            "error": "Evaluation failed: judge unavailable"
        }

        result = evaluate_one_question(
            collection=object(),
            question={
                "id": "q01",
                "mission": "apollo11",
                "category": "overview",
                "user_input": "Question",
                "reference": "Reference",
            },
            top_k=3,
            openai_api_key="test-key",
            generator_model="test-generator",
        )

        self.assertEqual(
            result["error"],
            "Evaluation failed: judge unavailable",
        )
        self.assertEqual(
            result["response"],
            "Generated answer.",
        )
        self.assertEqual(result["context_count"], 1)


class EvaluateBatchTests(unittest.TestCase):
    @patch("batch_evaluation.evaluate_one_question")
    def test_continues_after_one_question_fails(
        self,
        mocked_evaluate_one_question,
    ):
        collection = object()
        questions = [
            {
                "id": "q01",
                "mission": "apollo11",
                "category": "overview",
                "user_input": "First question",
                "reference": "First reference",
            },
            {
                "id": "q02",
                "mission": "apollo13",
                "category": "emergency",
                "user_input": "Second question",
                "reference": "Second reference",
            },
            {
                "id": "q03",
                "mission": "challenger",
                "category": "timeline",
                "user_input": "Third question",
                "reference": "Third reference",
            },
        ]
        first_result = {
            **questions[0],
            "response": "First answer",
            "faithfulness": 0.9,
        }
        third_result = {
            **questions[2],
            "response": "Third answer",
            "faithfulness": 0.8,
        }
        mocked_evaluate_one_question.side_effect = [
            first_result,
            RuntimeError("temporary API failure"),
            third_result,
        ]

        results = evaluate_batch(
            collection=collection,
            questions=questions,
            top_k=6,
            openai_api_key="test-key",
            generator_model="test-generator",
            evaluator_model="test-evaluator",
            embedding_model="test-embedding",
            openai_base_url="https://example.test/v1",
        )

        self.assertEqual(len(results), 3)
        self.assertIs(results[0], first_result)
        self.assertEqual(
            results[1]["error"],
            (
                "Question evaluation failed: "
                "RuntimeError: temporary API failure"
            ),
        )
        self.assertIsNone(results[1]["response"])
        self.assertEqual(
            results[1]["retrieved_contexts"],
            [],
        )
        self.assertIs(results[2], third_result)
        self.assertEqual(
            mocked_evaluate_one_question.call_count,
            3,
        )

        for question, call in zip(
            questions,
            mocked_evaluate_one_question.call_args_list,
        ):
            self.assertEqual(
                call.kwargs,
                {
                    "collection": collection,
                    "question": question,
                    "top_k": 6,
                    "openai_api_key": "test-key",
                    "generator_model": "test-generator",
                    "evaluator_model": "test-evaluator",
                    "embedding_model": "test-embedding",
                    "openai_base_url": "https://example.test/v1",
                },
            )

    @patch("batch_evaluation.evaluate_one_question")
    def test_rejects_empty_or_non_list_questions(
        self,
        mocked_evaluate_one_question,
    ):
        for questions in ([], "not a list", None):
            with self.subTest(questions=questions):
                with self.assertRaisesRegex(
                    ValueError,
                    "questions must be a non-empty list",
                ):
                    evaluate_batch(
                        collection=object(),
                        questions=questions,
                        top_k=3,
                        openai_api_key="test-key",
                        generator_model="test-generator",
                    )

        mocked_evaluate_one_question.assert_not_called()


class SummarizeBatchResultsTests(unittest.TestCase):
    def test_summarizes_valid_scores_and_tracks_failures(self):
        results = [
            {
                "id": "q01",
                "response_relevancy": 0.8,
                "faithfulness": 0.9,
                "context_precision": 0.7,
                "context_recall": 0.6,
                "factual_correctness": 0.75,
                "retrieval_f1": 0.646,
            },
            {
                "id": "q02",
                "response_relevancy": 0.6,
                "faithfulness": 0.7,
                "context_precision": float("nan"),
                "context_recall": 0.4,
                "factual_correctness": True,
                "retrieval_f1": 0.5,
            },
            {
                "id": "q03",
                "faithfulness": 0.0,
                "error": "Evaluation failed",
            },
        ]

        summary = summarize_batch_results(results)

        self.assertEqual(summary["question_count"], 3)
        self.assertEqual(
            summary["successful_question_count"],
            2,
        )
        self.assertEqual(summary["failed_question_count"], 1)
        self.assertEqual(
            summary["failed_question_ids"],
            ["q03"],
        )

        relevancy = summary["metrics"]["response_relevancy"]
        self.assertEqual(relevancy["count"], 2)
        self.assertAlmostEqual(relevancy["mean"], 0.7)
        self.assertEqual(relevancy["minimum"], 0.6)
        self.assertEqual(relevancy["maximum"], 0.8)

        faithfulness = summary["metrics"]["faithfulness"]
        self.assertEqual(faithfulness["count"], 2)
        self.assertAlmostEqual(faithfulness["mean"], 0.8)

        context_precision = summary["metrics"][
            "context_precision"
        ]
        self.assertEqual(context_precision["count"], 1)
        self.assertEqual(context_precision["mean"], 0.7)

        factual_correctness = summary["metrics"][
            "factual_correctness"
        ]
        self.assertEqual(factual_correctness["count"], 1)
        self.assertEqual(
            factual_correctness["mean"],
            0.75,
        )

        self.assertEqual(
            set(summary["metrics"]),
            set(EVALUATION_METRICS),
        )

    def test_rejects_empty_or_malformed_results(self):
        invalid_results = [
            [],
            "not a list",
            [None],
            [{"id": "q01"}, "not a dictionary"],
        ]

        for results in invalid_results:
            with self.subTest(results=results):
                with self.assertRaises(ValueError):
                    summarize_batch_results(results)


class RunBatchEvaluationTests(unittest.TestCase):
    @patch("batch_evaluation.summarize_batch_results")
    @patch("batch_evaluation.evaluate_batch")
    @patch("batch_evaluation.rag_client.initialize_rag_system")
    @patch("batch_evaluation.load_test_questions")
    def test_runs_complete_workflow_and_returns_report(
        self,
        mocked_load_questions,
        mocked_initialize_rag_system,
        mocked_evaluate_batch,
        mocked_summarize_results,
    ):
        dataset_path = Path("test_questions.json")
        questions = [
            {
                "id": "q01",
                "mission": "apollo11",
                "category": "overview",
                "user_input": "Question",
                "reference": "Reference",
            }
        ]
        collection = object()
        results = [
            {
                **questions[0],
                "response": "Answer",
                "faithfulness": 0.9,
            }
        ]
        summary = {
            "question_count": 1,
            "successful_question_count": 1,
            "failed_question_count": 0,
            "metrics": {
                "faithfulness": {
                    "count": 1,
                    "mean": 0.9,
                    "minimum": 0.9,
                    "maximum": 0.9,
                }
            },
        }
        mocked_load_questions.return_value = questions
        mocked_initialize_rag_system.return_value = (
            collection,
            True,
            None,
        )
        mocked_evaluate_batch.return_value = results
        mocked_summarize_results.return_value = summary

        report = run_batch_evaluation(
            dataset_path=dataset_path,
            chroma_dir="chroma_db",
            collection_name="nasa_missions",
            top_k=5,
            openai_api_key="secret-test-key",
            generator_model="test-generator",
            evaluator_model="test-evaluator",
            embedding_model="test-embedding",
            openai_base_url="https://example.test/v1",
        )

        mocked_load_questions.assert_called_once_with(
            dataset_path
        )
        mocked_initialize_rag_system.assert_called_once_with(
            chroma_dir="chroma_db",
            collection_name="nasa_missions",
            openai_api_key="secret-test-key",
            openai_base_url="https://example.test/v1",
            embedding_model="test-embedding",
        )
        mocked_evaluate_batch.assert_called_once_with(
            collection=collection,
            questions=questions,
            top_k=5,
            openai_api_key="secret-test-key",
            generator_model="test-generator",
            evaluator_model="test-evaluator",
            embedding_model="test-embedding",
            openai_base_url="https://example.test/v1",
        )
        mocked_summarize_results.assert_called_once_with(
            results
        )
        self.assertIs(report["results"], results)
        self.assertIs(report["summary"], summary)
        self.assertEqual(
            report["configuration"],
            {
                "dataset_path": "test_questions.json",
                "chroma_dir": "chroma_db",
                "collection_name": "nasa_missions",
                "top_k": 5,
                "generator_model": "test-generator",
                "evaluator_model": "test-evaluator",
                "embedding_model": "test-embedding",
            },
        )
        self.assertNotIn(
            "secret-test-key",
            json.dumps(report),
        )
        self.assertNotIn(
            "https://example.test/v1",
            json.dumps(report),
        )

    @patch("batch_evaluation.evaluate_batch")
    @patch("batch_evaluation.rag_client.initialize_rag_system")
    @patch("batch_evaluation.load_test_questions")
    def test_stops_when_collection_initialization_fails(
        self,
        mocked_load_questions,
        mocked_initialize_rag_system,
        mocked_evaluate_batch,
    ):
        mocked_load_questions.return_value = [
            {
                "id": "q01",
                "mission": "apollo11",
                "category": "overview",
                "user_input": "Question",
                "reference": "Reference",
            }
        ]
        mocked_initialize_rag_system.return_value = (
            None,
            False,
            "Collection unavailable",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Collection unavailable",
        ):
            run_batch_evaluation(
                dataset_path="test_questions.json",
                chroma_dir="chroma_db",
                collection_name="nasa_missions",
                top_k=3,
                openai_api_key="test-key",
                generator_model="test-generator",
            )

        mocked_evaluate_batch.assert_not_called()


class WriteBatchReportTests(unittest.TestCase):
    def test_writes_valid_utf8_json_and_replaces_non_finite_values(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "nested"
                / "evaluation-report.json"
            )
            report = {
                "summary": {
                    "mission": "阿波罗十一号",
                    "mean": float("nan"),
                },
                "results": [
                    {
                        "id": "q01",
                        "faithfulness": float("inf"),
                    },
                    {
                        "id": "q02",
                        "faithfulness": float("-inf"),
                    },
                ],
            }

            returned_path = write_batch_report(
                report=report,
                output_path=output_path,
            )

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.is_file())

            text = output_path.read_text(encoding="utf-8")
            parsed_report = json.loads(text)

            self.assertTrue(text.endswith("\n"))
            self.assertIn("阿波罗十一号", text)
            self.assertIsNone(
                parsed_report["summary"]["mean"]
            )
            self.assertIsNone(
                parsed_report["results"][0][
                    "faithfulness"
                ]
            )
            self.assertIsNone(
                parsed_report["results"][1][
                    "faithfulness"
                ]
            )

    def test_rejects_non_dictionary_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "report.json"

            with self.assertRaisesRegex(
                ValueError,
                "report must be a dictionary",
            ):
                write_batch_report(
                    report=["not a dictionary"],
                    output_path=output_path,
                )

            self.assertFalse(output_path.exists())


class BuildArgumentParserTests(unittest.TestCase):
    def test_uses_reproducible_project_defaults(self):
        arguments = build_argument_parser().parse_args([])

        self.assertEqual(
            arguments.dataset,
            "test_questions.json",
        )
        self.assertEqual(
            arguments.chroma_dir,
            "./chroma_db_openai",
        )
        self.assertEqual(
            arguments.collection_name,
            "nasa_space_missions_text",
        )
        self.assertEqual(arguments.top_k, 5)
        self.assertEqual(
            arguments.generator_model,
            "gpt-5-nano",
        )
        self.assertEqual(
            arguments.evaluator_model,
            "gpt-5-nano",
        )
        self.assertEqual(
            arguments.embedding_model,
            "text-embedding-3-small",
        )
        self.assertIsNone(arguments.openai_base_url)
        self.assertEqual(
            arguments.output,
            "evaluation_results.json",
        )

    def test_accepts_runtime_overrides(self):
        arguments = build_argument_parser().parse_args(
            [
                "--dataset",
                "custom-questions.json",
                "--chroma-dir",
                "custom-chroma",
                "--collection-name",
                "custom-collection",
                "--top-k",
                "9",
                "--generator-model",
                "generator-model",
                "--evaluator-model",
                "evaluator-model",
                "--embedding-model",
                "embedding-model",
                "--openai-base-url",
                "https://example.test/v1",
                "--output",
                "reports/custom.json",
            ]
        )

        self.assertEqual(
            arguments.dataset,
            "custom-questions.json",
        )
        self.assertEqual(
            arguments.chroma_dir,
            "custom-chroma",
        )
        self.assertEqual(
            arguments.collection_name,
            "custom-collection",
        )
        self.assertEqual(arguments.top_k, 9)
        self.assertEqual(
            arguments.generator_model,
            "generator-model",
        )
        self.assertEqual(
            arguments.evaluator_model,
            "evaluator-model",
        )
        self.assertEqual(
            arguments.embedding_model,
            "embedding-model",
        )
        self.assertEqual(
            arguments.openai_base_url,
            "https://example.test/v1",
        )
        self.assertEqual(
            arguments.output,
            "reports/custom.json",
        )


class PrintBatchSummaryTests(unittest.TestCase):
    def test_prints_question_scores_errors_and_aggregate(self):
        report = {
            "results": [
                {
                    "id": "q01",
                    "response_relevancy": 0.81234,
                    "faithfulness": 0.94567,
                },
                {
                    "id": "q02",
                    "error": "Generation failed",
                },
                {
                    "id": "q03",
                    "faithfulness": float("nan"),
                },
            ],
            "summary": {
                "question_count": 3,
                "successful_question_count": 2,
                "failed_question_count": 1,
            },
        }
        output = io.StringIO()

        with redirect_stdout(output):
            print_batch_summary(report)

        text = output.getvalue()
        self.assertIn("Per-question results:", text)
        self.assertIn(
            "- q01: response_relevancy=0.812, "
            "faithfulness=0.946",
            text,
        )
        self.assertIn(
            "- q02: ERROR - Generation failed",
            text,
        )
        self.assertIn(
            "- q03: no valid metric scores",
            text,
        )
        self.assertIn("Aggregate summary:", text)
        self.assertIn('"question_count": 3', text)
        self.assertIn(
            '"failed_question_count": 1',
            text,
        )

    def test_rejects_malformed_report_sections(self):
        invalid_reports = [
            {},
            {"results": "not a list", "summary": {}},
            {"results": [], "summary": "not a dictionary"},
        ]

        for report in invalid_reports:
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    print_batch_summary(report)


class MainTests(unittest.TestCase):
    def test_loads_environment_runs_batch_writes_and_prints(self):
        report = {
            "results": [],
            "summary": {
                "question_count": 0,
            },
        }
        output_path = Path("reports/evaluation.json")

        with (
            patch.dict(
                "batch_evaluation.os.environ",
                {
                    "OPENAI_API_KEY": "  test-key  ",
                    "OPENAI_BASE_URL": (
                        "  https://environment.test/v1  "
                    ),
                },
                clear=True,
            ),
            patch(
                "batch_evaluation.load_dotenv"
            ) as mocked_load_dotenv,
            patch(
                "batch_evaluation.run_batch_evaluation",
                return_value=report,
            ) as mocked_run_batch,
            patch(
                "batch_evaluation.write_batch_report",
                return_value=output_path,
            ) as mocked_write_report,
            patch(
                "batch_evaluation.print_batch_summary"
            ) as mocked_print_summary,
            redirect_stdout(io.StringIO()) as output,
        ):
            main(
                [
                    "--dataset",
                    "custom-questions.json",
                    "--chroma-dir",
                    "custom-chroma",
                    "--collection-name",
                    "custom-collection",
                    "--top-k",
                    "7",
                    "--generator-model",
                    "generator-model",
                    "--evaluator-model",
                    "evaluator-model",
                    "--embedding-model",
                    "embedding-model",
                    "--output",
                    str(output_path),
                ]
            )

        project_root = Path(__file__).resolve().parents[1]
        mocked_load_dotenv.assert_called_once_with(
            dotenv_path=project_root / ".env"
        )
        mocked_run_batch.assert_called_once_with(
            dataset_path="custom-questions.json",
            chroma_dir="custom-chroma",
            collection_name="custom-collection",
            top_k=7,
            openai_api_key="test-key",
            generator_model="generator-model",
            evaluator_model="evaluator-model",
            embedding_model="embedding-model",
            openai_base_url="https://environment.test/v1",
        )
        mocked_write_report.assert_called_once_with(
            report=report,
            output_path=str(output_path),
        )
        mocked_print_summary.assert_called_once_with(report)
        self.assertIn(
            f"Report written to: {output_path}",
            output.getvalue(),
        )

    def test_rejects_missing_api_key_before_evaluation(self):
        with (
            patch.dict(
                "batch_evaluation.os.environ",
                {},
                clear=True,
            ),
            patch("batch_evaluation.load_dotenv"),
            patch(
                "batch_evaluation.run_batch_evaluation"
            ) as mocked_run_batch,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                main([])

        self.assertEqual(raised.exception.code, 2)
        mocked_run_batch.assert_not_called()

    def test_rejects_non_positive_top_k_before_evaluation(self):
        with (
            patch.dict(
                "batch_evaluation.os.environ",
                {"OPENAI_API_KEY": "test-key"},
                clear=True,
            ),
            patch("batch_evaluation.load_dotenv"),
            patch(
                "batch_evaluation.run_batch_evaluation"
            ) as mocked_run_batch,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                main(["--top-k", "0"])

        self.assertEqual(raised.exception.code, 2)
        mocked_run_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
