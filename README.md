# NASA Mission Intelligence RAG

This project implements a complete Retrieval-Augmented Generation (RAG)
workflow for NASA mission documents about Apollo 11, Apollo 13, and
Challenger. It extends the Udacity starter project with a reproducible Python
environment, conservative OCR and transcript cleaning, configurable
token-based chunking, persistent OpenAI embeddings in ChromaDB, grounded chat,
and real-time and batch RAGAS evaluation.

The source corpus contains OCR artifacts, page furniture, tables, charts, and
transcript formatting that are not all useful as retrieval text. The data
pipeline therefore cleans each source while preserving useful narrative
content and provenance before creating embeddings.

## Architecture

```text
Ingestion
data_text/
    -> nasa_text_cleaners.py
       cleaned semantic records + source metadata
    -> embedding_pipeline.py
       aggregate by source file -> token chunks -> OpenAI embeddings
    -> persistent ChromaDB collection

Interactive workflow
chat.py
    -> rag_client.py
       semantic top-k retrieval -> deduplicated, source-attributed context
    -> llm_client.py
       grounded prompt + conversation history -> answer
    -> ragas_evaluator.py
       real-time metrics

Independent batch workflow
test_questions.json
    -> batch_evaluation.py
       retrieval -> generation -> reference-based RAGAS metrics
    -> JSON evaluation report
```

## Rubric coverage

### Embedding and data pipeline

- `chunk_size` and `chunk_overlap` are configurable at runtime.
- Every chunk is checked against the configured token limit.
- Consecutive chunks use consistent overlap.
- OpenAI embeddings are generated for every stored chunk.
- Every chunk carries metadata including mission, source, and filepath.
- Existing documents support `skip`, `update`, and `replace` update modes.
- ChromaDB persistence directory and collection name are configurable.
- `--stats-only` reports the collection name, stored chunk/document count,
  source-file count, and mission/type aggregates.

### Retrieval and LLM integration

- Questions are embedded and queried against the persistent ChromaDB
  collection.
- Retrieval `top_k` is configurable, with optional mission metadata filtering.
- Results are ranked, deduplicated, and formatted with clear source
  attribution.
- The NASA expert prompt requires citations, reliance on retrieved evidence,
  and explicit uncertainty when the evidence is insufficient.
- Conversation history is maintained as role/content turns while retrieved
  context is supplied only for the current request.

### Evaluation

- Interactive evaluation measures response relevancy and faithfulness.
- Reference-based batch evaluation also measures context precision, context
  recall, factual correctness, and derived retrieval F1.
- The batch workflow loads the supplied test set, evaluates each question, and
  writes both per-question results and aggregate means.
- Malformed inputs and individual metric failures return structured errors
  instead of terminating the whole evaluation run.

## Project structure

```text
.
├── data_text/                  # NASA OCR and transcript source files
├── tests/                      # Unit and integration-oriented tests
├── nasa_text_cleaners.py       # Source-aware OCR/transcript cleaning
├── embedding_pipeline.py       # Chunking, embedding, ChromaDB persistence, CLI
├── rag_client.py               # Semantic retrieval and context construction
├── llm_client.py               # Grounded OpenAI response generation
├── ragas_evaluator.py          # Async RAGAS metric evaluation
├── chat.py                     # Streamlit chat interface
├── batch_evaluation.py         # End-to-end test-set evaluation CLI
├── test_questions.json         # 17 mission questions with reference answers
├── evaluation_results_2026-07-28.json
│                                # Reproducible full-evaluation evidence
├── submission.ipynb            # Exploratory project notebook
├── pyproject.toml              # Project metadata and dependencies
└── uv.lock                     # Locked dependency versions
```

## Requirements and installation

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI-compatible API endpoint and API key

Install the locked environment:

```bash
uv sync
```

`uv sync` creates or updates `.venv` and installs the exact dependency set
recorded in `uv.lock`, including the development test dependencies.

Create a local `.env` file:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_compatible_endpoint
```

`OPENAI_BASE_URL` is optional when using the standard OpenAI endpoint. For the
Streamlit app, the following optional environment variables override the
default generation and judge models:

```dotenv
OPENAI_CHAT_MODEL=gpt-5.4-nano
OPENAI_EVALUATOR_MODEL=gpt-5.4-mini
```

The batch workflow instead uses the explicit `--generator-model` and
`--evaluator-model` CLI flags shown below.

Do not commit `.env`; it is ignored by Git.

## 1. Test the implementation

```bash
uv run python -m pytest -q
```

This runs the complete test suite inside the locked project environment.

## 2. Build or update the vector collection

```bash
uv run python embedding_pipeline.py \
  --data-path data_text \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --embedding-model text-embedding-3-small \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --batch-size 50 \
  --update-mode replace
```

The important runtime options are:

| Option | Meaning |
|---|---|
| `--chunk-size` | Maximum number of tokens allowed in a chunk |
| `--chunk-overlap` | Target number of repeated tokens between neighboring chunks |
| `--batch-size` | Number of chunks sent in one embedding/storage batch |
| `--update-mode skip` | Embed only chunk IDs not already in the collection |
| `--update-mode update` | Add new IDs and overwrite matching existing IDs |
| `--update-mode replace` | Synchronize each source by upserting current chunks and deleting stale chunks |

Use `replace` after cleaning or chunking rules change, because the new chunk set
may no longer have the same IDs as the old set. Use `skip` for a safe,
incremental rerun when the source and chunking configuration have not changed.

To inspect an existing collection without rebuilding it:

```bash
uv run python embedding_pipeline.py \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --stats-only
```

Initialization still validates the API configuration, but `--stats-only` does
not create new embeddings.

## 3. Run the chat application

```bash
uv run streamlit run chat.py
```

The Streamlit sidebar controls the ChromaDB backend, collection, retrieval
count, optional mission filter, generator model, and whether evaluation is
enabled. Generated answers are instructed to cite numbered context documents,
and the interface displays real-time metrics when evaluation is enabled. The
evaluator model is configured through `OPENAI_EVALUATOR_MODEL`; a separate
retrieved-source panel is not currently rendered.

## 4. Run the full batch evaluation

```bash
uv run python batch_evaluation.py \
  --dataset test_questions.json \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --top-k 5 \
  --generator-model gpt-5.4-nano \
  --evaluator-model gpt-5.4-mini \
  --embedding-model text-embedding-3-small \
  --output evaluation_results.json
```

This is a live API workflow: it performs retrieval, answer generation, and
model-based evaluation for every test question, so it consumes API tokens and
can take several minutes.

### Metric meanings

| Metric | What it asks |
|---|---|
| Response relevancy | Does the answer address the question? |
| Faithfulness | Are answer claims supported by the retrieved context? |
| Context precision | How much of the retrieved context is useful for the reference answer? |
| Context recall | How much of the reference answer is covered by retrieved context? |
| Factual correctness | How well do answer claims agree with the reference answer? |
| Retrieval F1 | Harmonic mean of context precision and context recall |

All scores range from 0 to 1, where higher is better, but there is no universal
pass threshold. Response relevancy compares the question with questions
reconstructed from the answer. Faithfulness checks whether answer claims are
supported by retrieved text. Context precision rewards useful chunks appearing
early in the ranking, while context recall checks whether retrieved text
contains the claims needed by the reference answer. Factual correctness is
claim-level F1 between the generated and reference answers. Retrieval F1 is
this project's harmonic mean of context precision and context recall.

### Concise metric example

Suppose an Apollo 13 question asks what happened to the oxygen and fuel-cell
systems and what decision followed:

1. The first retrieved chunk describes the tank and fuel-cell failures
   (`useful`).
2. The second contains only launch-time information (`not useful`).
3. The third says the landing was scrubbed and the lunar module became a
   lifeboat (`useful`).

The answer repeats those supported facts, but the reference also says the
mission continued as a lunar-flyby abort. Under these simplified judge
decisions:

- response relevancy would be high because the answer directly addresses the
  question;
- faithfulness would be `1.00` if every answer claim is supported;
- context precision would be `(1 + 2/3) / 2 = 0.833`, because useful chunks
  occur at ranks 1 and 3;
- context recall would be `4/5 = 0.80` if four of five reference claims are
  supported by the retrieved text;
- factual correctness would be about `0.89` for four matching claims, no false
  answer claim, and one missing reference claim;
- retrieval F1 would be approximately `0.816`, the harmonic mean of `0.833`
  and `0.80`.

These numbers are illustrative, not observed results. RAGAS uses an LLM to
generate questions, split claims, and make support judgments, so the actual
decomposition and scores can vary between runs.

RAGAS metrics are model-based and may vary slightly between runs. If one
metric fails, its value is marked as `NaN` internally and serialized as JSON
`null` with a metric-specific error, while the remaining metrics and questions
continue.

## Verified baseline

The local collection used for the baseline contains:

- 5,584 chunks from 12 source files
- Apollo 11: 2,707 chunks
- Apollo 13: 2,526 chunks
- Challenger: 351 chunks
- 500-token maximum chunk size and 100-token configured overlap
- 1,536-dimensional `text-embedding-3-small` vectors

The 2026-07-28 full run completed all 17 test questions without a
question-level failure. Its configuration and complete per-question evidence
are stored in
[`evaluation_results_2026-07-28.json`](evaluation_results_2026-07-28.json).

| Metric | Mean |
|---|---:|
| Response relevancy | 0.7846 |
| Faithfulness | 0.8158 |
| Context precision | 0.5560 |
| Context recall | 0.5000 |
| Factual correctness | 0.2953 |
| Retrieval F1 | 0.4300 |

The metric means answer different questions:

- `0.7846` response relevancy: answers generally address the questions.
- `0.8158` faithfulness: most generated claims are supported by retrieved
  evidence.
- `0.5560` context precision: useful evidence is only moderately concentrated
  near the top of the retrieval ranking.
- `0.5000` context recall: retrieval covers only about half of the
  reference-answer claims under the evaluator's attribution judgments.
- `0.2953` factual correctness: generated and reference claims align weakly
  overall, even though answers are usually grounded in what was retrieved.
- `0.4300` retrieval F1: the combined precision/recall balance is
  moderate-to-low.

Together, these results suggest that answer grounding is stronger than
retrieval completeness. The main improvement target is therefore to retrieve
more complete, reference-relevant evidence and then reduce missing or
mismatched answer claims. These results are a baseline, not a claim that
retrieval is fully optimized.

## Data-cleaning approach

`nasa_text_cleaners.py` applies source-aware rules rather than one destructive
generic cleanup pass. It aims to:

- remove repeated page furniture and obvious OCR noise;
- preserve readable narrative, transcript turns, and useful technical text;
- avoid embedding unusable chart or table fragments;
- retain mission, source type, filepath, page/section information, and raw
  provenance needed to audit a retrieved chunk.

The cleaners return cleaned report blocks or transcript turns.
`embedding_pipeline.py` then aggregates those semantic records by source file
before chunking, so overlap never crosses file boundaries. Cleaning happens
before chunking because embeddings cannot repair missing content or distinguish
meaningful text from severe OCR artifacts after the two have already been
mixed together.

## Known limitations

- Conservative cleaning cannot remove every OCR or ASR error without also
  risking loss of useful mission content.
- Exact text deduplication does not remove every near-duplicate passage created
  by overlap or repeated source material.
- RAGAS is a model-as-judge evaluation; results are stochastic and live runs
  incur API cost.
- Reference-answer metrics are useful diagnostics but are sensitive to the
  wording and coverage of the supplied references.
- The system prompt requires source citations, but there is not yet a separate
  automatic citation-entailment verifier.

## Original project

This implementation is based on Udacity's
[NASA Mission Intelligence starter project](https://github.com/udacity/cd13318-exercises-project/tree/main/Project-NASA-Mission-Intelligence-Starter).
