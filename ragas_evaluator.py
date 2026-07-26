import os
import asyncio
from typing import Any, Dict, List, Optional
from openai import OpenAI

try:
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        FactualCorrectness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

async def _score_metrics_async(
    scorers: Dict[str, Any],
    metric_inputs: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    metric_names = list(scorers)
    metric_results = await asyncio.gather(
        *(
            scorers[metric_name].ascore(
                **metric_inputs[metric_name]
            )
            for metric_name in metric_names
        )
    )
    return {
        metric_name: float(metric_result.value)
        for metric_name, metric_result in zip(
            metric_names,
            metric_results,
        )
    }

def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    evaluator_model: str = "gpt-5-nano",
    reference: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
) -> Dict[str, float | str]:
    """Evaluate a question, retrieved contexts, and model answer."""
    if not isinstance(question, str) or not question.strip():
        return {"error": "question must be a non-empty string"}
    if not isinstance(answer, str) or not answer.strip():
        return {"error": "answer must be a non-empty string"}
    if not isinstance(contexts, list) or not contexts:
        return {"error": "contexts must be a non-empty list"}
    if reference is not None and (
        not isinstance(reference, str) or not reference.strip()
    ):
        return {
            "error": "reference must be a non-empty string when provided"
        }
    if any(
        not isinstance(context, str) or not context.strip()
        for context in contexts
    ):
        return {"error": "each context must be a non-empty string"}
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS is not available"}
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
    if not isinstance(api_key, str) or not api_key.strip():
        return {"error": "OpenAI API key must not be empty"}
    if not isinstance(evaluator_model, str) or not evaluator_model.strip():
        return {"error": "evaluator_model must not be empty"}
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        return {"error": "embedding_model must not be empty"}
    normalized_base_url = (
        base_url.strip()
        if isinstance(base_url, str) and base_url.strip()
        else None
    )
    try:
        client = OpenAI(
            api_key=api_key.strip(),
            base_url=normalized_base_url,
        )
        evaluator_llm = llm_factory(
            model=evaluator_model.strip(),
            provider="openai",
            client=client,
        )
        evaluator_embeddings = OpenAIEmbeddings(
            client=client,
            model=embedding_model.strip(),
        )
        scorers = {
            "response_relevancy": AnswerRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            ),
            "faithfulness": Faithfulness(
                llm=evaluator_llm,
            ),
        }
        if reference is not None:
            scorers.update(
                {
                    "context_precision": ContextPrecision(
                        llm=evaluator_llm,
                    ),
                    "context_recall": ContextRecall(
                        llm=evaluator_llm,
                    ),
                    "factual_correctness": FactualCorrectness(
                        llm=evaluator_llm,
                        mode="f1",
                    ),
                }
            )
    except Exception as error:
        return {
            "error": (
                f"Evaluation setup failed: "
                f"{type(error).__name__}: {error}"
            )
        }
    cleaned_question = question.strip()
    cleaned_answer = answer.strip()
    cleaned_contexts = [
        context.strip()
        for context in contexts
    ]
    metric_inputs = {
        "response_relevancy": {
            "user_input": cleaned_question,
            "response": cleaned_answer,
        },
        "faithfulness": {
            "user_input": cleaned_question,
            "response": cleaned_answer,
            "retrieved_contexts": cleaned_contexts,
        },
    }
    if reference is not None:
        cleaned_reference = reference.strip()
        metric_inputs.update(
            {
                "context_precision": {
                    "user_input": cleaned_question,
                    "reference": cleaned_reference,
                    "retrieved_contexts": cleaned_contexts,
                },
                "context_recall": {
                    "user_input": cleaned_question,
                    "reference": cleaned_reference,
                    "retrieved_contexts": cleaned_contexts,
                },
                "factual_correctness": {
                    "response": cleaned_answer,
                    "reference": cleaned_reference,
                },
            }
        )
    try:
        scores = asyncio.run(
            _score_metrics_async(
                scorers=scorers,
                metric_inputs=metric_inputs,
            )
        )
    except Exception as error:
        return {
            "error": (
                f"Evaluation failed: "
                f"{type(error).__name__}: {error}"
            )
        }
    if reference is not None:
        precision = scores["context_precision"]
        recall = scores["context_recall"]
        denominator = precision + recall

        scores["retrieval_f1"] = (
            0.0
            if denominator == 0
            else 2 * precision * recall / denominator
        )
    return scores
