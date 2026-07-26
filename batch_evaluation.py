import os
import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

import llm_client
import rag_client
import ragas_evaluator

EVALUATION_METRICS = (
    "response_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
    "factual_correctness",
    "retrieval_f1",
)

REQUIRED_QUESTION_FIELDS = {
    "id",
    "mission",
    "category",
    "user_input",
    "reference",
}

def load_test_questions(
    dataset_path: str | Path,
) -> List[Dict[str, Any]]:
    """Load and validate the batch-evaluation questions."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset does not exist: {path}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Evaluation dataset is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            "Evaluation dataset must be an object"
        )
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "Evaluation dataset must contain a non-empty questions list"
        )
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(
                f"Question at index {index} must be an object"
            )
        missing_fields = (
            REQUIRED_QUESTION_FIELDS - question.keys()
        )
        if missing_fields:
            raise ValueError(
                f"Question at index {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )
        for field_name in sorted(
            REQUIRED_QUESTION_FIELDS
        ):
            field_value = question[field_name]
            if (
                not isinstance(field_value, str)
                or not field_value.strip()
            ):
                raise ValueError(
                    f"Question at index {index} field "
                    f"'{field_name}' must be a non-empty string"
                )
    return questions

def retrieve_question_evidence(
    collection: Any,
    question: Dict[str, Any],
    top_k: int,
) -> tuple[List[str], List[Dict[str, Any]], str]:
    """Retrieve evidence for one evaluation question."""
    retrieval_result = rag_client.retrieve_documents(
        collection=collection,
        query=question["user_input"],
        n_results=top_k,
        mission_filter=question["mission"],
    )
    try:
        documents = retrieval_result["documents"][0]
        metadatas = retrieval_result["metadatas"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "Retrieval returned malformed results"
        ) from error
    if not isinstance(documents, list):
        raise RuntimeError(
            "Retrieved documents must be a list"
        )
    if not isinstance(metadatas, list):
        raise RuntimeError(
            "Retrieved metadatas must be a list"
        )
    if not documents:
        raise RuntimeError(
            "Retrieval returned no documents"
        )
    formatted_context = rag_client.format_context(
        documents=documents,
        metadatas=metadatas,
    )
    return documents, metadatas, formatted_context

def generate_answer_for_question(
    collection: Any,
    question: Dict[str, Any],
    top_k: int,
    openai_api_key: str,
    generator_model: str,
    openai_base_url: Optional[str] = None,
) -> tuple[
    List[str],
    List[Dict[str, Any]],
    str,
]:
    """Retrieve evidence and generate one grounded answer."""
    (
        documents,
        metadatas,
        formatted_context,
    ) = retrieve_question_evidence(
        collection=collection,
        question=question,
        top_k=top_k,
    )
    answer = llm_client.generate_response(
        openai_key=openai_api_key,
        user_message=question["user_input"],
        context=formatted_context,
        conversation_history=[],
        model=generator_model,
        openai_base_url=openai_base_url,
    )
    return documents, metadatas, answer

def evaluate_one_question(
    collection: Any,
    question: Dict[str, Any],
    top_k: int,
    openai_api_key: str,
    generator_model: str,
    evaluator_model: str = "gpt-5-nano",
    embedding_model: str = "text-embedding-3-small",
    openai_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate and evaluate one batch question."""
    (
        documents,
        metadatas,
        answer,
    ) = generate_answer_for_question(
        collection=collection,
        question=question,
        top_k=top_k,
        openai_api_key=openai_api_key,
        generator_model=generator_model,
        openai_base_url=openai_base_url,
    )

    scores = ragas_evaluator.evaluate_response_quality(
        question=question["user_input"],
        answer=answer,
        contexts=documents,
        evaluator_model=evaluator_model,
        reference=question["reference"],
        embedding_model=embedding_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    return {
        **question,
        "response": answer,
        "retrieved_contexts": documents,
        "retrieved_metadatas": metadatas,
        "context_count": len(documents),
        **scores,
    }

def evaluate_batch(
    collection: Any,
    questions: List[Dict[str, Any]],
    top_k: int,
    openai_api_key: str,
    generator_model: str,
    evaluator_model: str = "gpt-5-nano",
    embedding_model: str = "text-embedding-3-small",
    openai_base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Evaluate every question without one failure stopping the batch."""
    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "questions must be a non-empty list"
        )
    results = []
    for question in questions:
        try:
            result = evaluate_one_question(
                collection=collection,
                question=question,
                top_k=top_k,
                openai_api_key=openai_api_key,
                generator_model=generator_model,
                evaluator_model=evaluator_model,
                embedding_model=embedding_model,
                openai_base_url=openai_base_url,
            )
        except Exception as error:
            result = {
                **question,
                "response": None,
                "retrieved_contexts": [],
                "retrieved_metadatas": [],
                "context_count": 0,
                "error": (
                    f"Question evaluation failed: "
                    f"{type(error).__name__}: {error}"
                ),
            }
        results.append(result)
    return results

def summarize_batch_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize successes, failures, and metric distributions."""
    if not isinstance(results, list) or not results:
        raise ValueError(
            "results must be a non-empty list"
        )
    if any(
        not isinstance(result, dict)
        for result in results
    ):
        raise ValueError(
            "each result must be a dictionary"
        )
    failed_results = [
        result
        for result in results
        if "error" in result
    ]
    metric_summaries = {}
    for metric_name in EVALUATION_METRICS:
        values = []
        for result in results:
            value = result.get(metric_name)
            if (
                "error" not in result
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                values.append(float(value))
        if values:
            metric_summaries[metric_name] = {
                "count": len(values),
                "mean": fmean(values),
                "minimum": min(values),
                "maximum": max(values),
            }
    return {
        "question_count": len(results),
        "successful_question_count": (
            len(results) - len(failed_results)
        ),
        "failed_question_count": len(failed_results),
        "failed_question_ids": [
            result.get("id", "unknown")
            for result in failed_results
        ],
        "metrics": metric_summaries,
    }

def run_batch_evaluation(
    dataset_path: str | Path,
    chroma_dir: str,
    collection_name: str,
    top_k: int,
    openai_api_key: str,
    generator_model: str,
    evaluator_model: str = "gpt-5-nano",
    embedding_model: str = "text-embedding-3-small",
    openai_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete batch-evaluation workflow."""
    questions = load_test_questions(dataset_path)
    (
        collection,
        initialized,
        initialization_error,
    ) = rag_client.initialize_rag_system(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        embedding_model=embedding_model,
    )
    if not initialized or collection is None:
        raise RuntimeError(
            initialization_error
            or "Failed to initialize the RAG collection"
        )
    results = evaluate_batch(
        collection=collection,
        questions=questions,
        top_k=top_k,
        openai_api_key=openai_api_key,
        generator_model=generator_model,
        evaluator_model=evaluator_model,
        embedding_model=embedding_model,
        openai_base_url=openai_base_url,
    )
    summary = summarize_batch_results(results)
    return {
        "configuration": {
            "dataset_path": str(dataset_path),
            "chroma_dir": chroma_dir,
            "collection_name": collection_name,
            "top_k": top_k,
            "generator_model": generator_model,
            "evaluator_model": evaluator_model,
            "embedding_model": embedding_model,
        },
        "summary": summary,
        "results": results,
    }

def _make_json_safe(value: Any) -> Any:
    """Replace non-finite floats with JSON null values."""
    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None
    if isinstance(value, dict):
        return {
            key: _make_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _make_json_safe(item)
            for item in value
        ]
    return value

def write_batch_report(
    report: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write per-question results and aggregates as valid JSON."""
    if not isinstance(report, dict):
        raise ValueError(
            "report must be a dictionary"
        )
    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    safe_report = _make_json_safe(report)
    path.write_text(
        json.dumps(
            safe_report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

def build_argument_parser() -> argparse.ArgumentParser:
    """Build runtime configuration for batch evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end evaluation of the NASA RAG system"
        )
    )
    parser.add_argument(
        "--dataset",
        default="test_questions.json",
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
    )
    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--generator-model",
        default="gpt-5-nano",
    )
    parser.add_argument(
        "--evaluator-model",
        default="gpt-5-nano",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
    )
    return parser

def print_batch_summary(
    report: Dict[str, Any],
) -> None:
    """Print per-question scores and aggregate metrics."""
    results = report.get("results")
    summary = report.get("summary")
    if not isinstance(results, list):
        raise ValueError(
            "report results must be a list"
        )
    if not isinstance(summary, dict):
        raise ValueError(
            "report summary must be a dictionary"
        )
    print("Per-question results:")
    for result in results:
        question_id = result.get("id", "unknown")

        if "error" in result:
            print(
                f"- {question_id}: ERROR - "
                f"{result['error']}"
            )
            continue
        metric_parts = []
        for metric_name in EVALUATION_METRICS:
            value = result.get(metric_name)

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                metric_parts.append(
                    f"{metric_name}={value:.3f}"
                )
        metric_text = (
            ", ".join(metric_parts)
            or "no valid metric scores"
        )
        print(f"- {question_id}: {metric_text}")
    print("\nAggregate summary:")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )

def main(
    argv: Optional[List[str]] = None,
) -> None:
    """Run batch evaluation from command-line arguments."""
    project_root = Path(__file__).resolve().parent
    load_dotenv(
        dotenv_path=project_root / ".env"
    )
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    if arguments.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    api_key = os.getenv("OPENAI_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        parser.error(
            "OPENAI_API_KEY must be set in .env "
            "or the environment"
        )
    base_url = (
        arguments.openai_base_url
        or os.getenv("OPENAI_BASE_URL")
    )
    if isinstance(base_url, str):
        base_url = base_url.strip() or None
    report = run_batch_evaluation(
        dataset_path=arguments.dataset,
        chroma_dir=arguments.chroma_dir,
        collection_name=arguments.collection_name,
        top_k=arguments.top_k,
        openai_api_key=api_key.strip(),
        generator_model=arguments.generator_model,
        evaluator_model=arguments.evaluator_model,
        embedding_model=arguments.embedding_model,
        openai_base_url=base_url,
    )
    output_path = write_batch_report(
        report=report,
        output_path=arguments.output,
    )
    print_batch_summary(report)
    print(f"\nReport written to: {output_path}")

if __name__ == "__main__":
    main()