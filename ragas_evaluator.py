import os
import asyncio
import math
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

DEFAULT_EVALUATOR_MODEL = "gpt-5.4-mini"
EVALUATOR_MAX_TOKENS = 4096

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

def _reasoning_effort_for_model(
    model_name: str,
) -> Optional[str]:
    """Return a supported low-cost reasoning effort when known."""
    normalized_model = model_name.casefold()
    if normalized_model.startswith("gpt-5."):
        return "low"
    if (
        normalized_model == "gpt-5"
        or normalized_model.startswith("gpt-5-")
    ):
        return "minimal"
    return None

def _normalize_ragas_openai_model_args(
    evaluator_llm: Any,
    model_name: str,
    max_tokens: int,
    reasoning_effort: Optional[str],
) -> None:
    """Repair dotted GPT-5 request arguments for RAGAS 0.4.3."""
    if not model_name.casefold().startswith("gpt-5."):
        return
    model_args = getattr(evaluator_llm, "model_args", None)
    if not isinstance(model_args, dict):
        raise TypeError(
            "RAGAS evaluator LLM must expose model_args"
        )

    # RAGAS 0.4.3 tries int("5.4"), misses the reasoning-model
    # branch, and otherwise sends unsupported legacy parameters.
    model_args.pop("max_tokens", None)
    model_args.pop("top_p", None)
    model_args["temperature"] = 1.0
    model_args["max_completion_tokens"] = max_tokens
    if reasoning_effort is not None:
        model_args["reasoning_effort"] = reasoning_effort

async def _score_metrics_async(
    scorers: Dict[str, Any],
    metric_inputs: Dict[str, Dict[str, Any]],
) -> Dict[str, float | str]:
    metric_names = list(scorers)
    metric_results = await asyncio.gather(
        *(
            scorers[metric_name].ascore(
                **metric_inputs[metric_name]
            )
            for metric_name in metric_names
        ),
        return_exceptions=True,
    )
    scores: Dict[str, float | str] = {}
    for metric_name, metric_result in zip(
        metric_names,
        metric_results,
    ):
        if isinstance(metric_result, BaseException):
            scores[metric_name] = math.nan
            scores[f"{metric_name}_error"] = (
                f"{type(metric_result).__name__}: "
                f"{metric_result}"
            )
            continue
        try:
            scores[metric_name] = float(metric_result.value)
        except (AttributeError, TypeError, ValueError) as error:
            scores[metric_name] = math.nan
            scores[f"{metric_name}_error"] = (
                f"{type(error).__name__}: {error}"
            )
    return scores

async def _evaluate_metrics_with_client_async(
    api_key: str,
    base_url: Optional[str],
    evaluator_model: str,
    embedding_model: str,
    llm_options: Dict[str, Any],
    reasoning_effort: Optional[str],
    include_reference_metrics: bool,
    metric_inputs: Dict[str, Dict[str, Any]],
) -> Dict[str, float | str]:
    """Create, use, and close the async client on one event loop."""
    client: Optional[AsyncOpenAI] = None
    try:
        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            evaluator_llm = llm_factory(
                model=evaluator_model,
                provider="openai",
                client=client,
                **llm_options,
            )
            _normalize_ragas_openai_model_args(
                evaluator_llm=evaluator_llm,
                model_name=evaluator_model,
                max_tokens=EVALUATOR_MAX_TOKENS,
                reasoning_effort=reasoning_effort,
            )
            evaluator_embeddings = OpenAIEmbeddings(
                client=client,
                model=embedding_model,
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
            if include_reference_metrics:
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
        try:
            return await _score_metrics_async(
                scorers=scorers,
                metric_inputs=metric_inputs,
            )
        except Exception as error:
            return {
                "error": (
                    f"Evaluation failed: "
                    f"{type(error).__name__}: {error}"
                )
            }
    finally:
        if client is not None:
            await client.close()

def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
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
    normalized_evaluator_model = evaluator_model.strip()
    reasoning_effort = _reasoning_effort_for_model(
        normalized_evaluator_model
    )
    llm_options: Dict[str, Any] = {
        "max_tokens": EVALUATOR_MAX_TOKENS
    }
    if reasoning_effort is not None:
        llm_options["reasoning_effort"] = reasoning_effort
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
            _evaluate_metrics_with_client_async(
                api_key=api_key.strip(),
                base_url=normalized_base_url,
                evaluator_model=normalized_evaluator_model,
                embedding_model=embedding_model.strip(),
                llm_options=llm_options,
                reasoning_effort=reasoning_effort,
                include_reference_metrics=reference is not None,
                metric_inputs=metric_inputs,
            )
        )
    except Exception as error:
        return {
            "error": (
                f"Evaluation lifecycle failed: "
                f"{type(error).__name__}: {error}"
            )
        }
    if "error" in scores:
        return scores
    if reference is not None:
        precision = scores.get("context_precision")
        recall = scores.get("context_recall")
        if (
            isinstance(precision, (int, float))
            and not isinstance(precision, bool)
            and isinstance(recall, (int, float))
            and not isinstance(recall, bool)
            and math.isfinite(precision)
            and math.isfinite(recall)
        ):
            denominator = precision + recall
            scores["retrieval_f1"] = (
                0.0
                if denominator == 0
                else 2 * precision * recall / denominator
            )
        else:
            scores["retrieval_f1"] = math.nan
            scores["retrieval_f1_error"] = (
                "Context precision and context recall must both "
                "succeed before retrieval F1 can be calculated"
            )
    return scores
