# Jev as a Decision Layer

Python examples comparing Jev and OpenAI for structured decisions, routing, and evaluation using synthetic cases.

| Script | Example |
| --- | --- |
| `model_routing.py` | Model routing policy decisions |
| `guardrails.py` | Prompt injection detection |
| `tool_call_gating.py` | Dry-run tool-call gating |
| `inbox_triage.py` | Inbox triage with synthetic emails |
| `reranking.py` | Passage relevance scoring and reranking |
| `llm_evaluations.py` | Reference-grounded answer evaluation |
| `bulk_labeling.py` | Support-ticket labeling |
| `realtime_control.py` | Game action selection and deadline checks |
| `confidence_gating.py` | Offline replay of labeling results through review gates |
| `check_api_connection.py` | Unauthenticated HTTPS connectivity diagnostics |

## Setup

Use Python 3.10 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Set `TYPESAFE_API_KEY` and `OPENAI_API_KEY` in `.env`. Live benchmark scripts load the file beside the script, preserve existing environment variables, and prompt for missing keys. The local `.env` file and generated result folders are excluded from Git.

## Run examples

Each benchmark has an offline self-test that requires no API keys:

```sh
python3 model_routing.py --self-test
python3 bulk_labeling.py --self-test
python3 confidence_gating.py --self-test
```

Run a live comparison with a small case limit:

```sh
python3 model_routing.py --limit 2 --repeats 1
python3 bulk_labeling.py --limit 2 --repeats 1
```

Live benchmarks make paid API requests and save JSON/JSONL results in timestamped output folders. Use `--help` on a benchmark to see model, output, and case options. Models and cost estimates are defined in each script; check them before interpreting benchmark costs.

Replay saved bulk-labeling results without additional API calls:

```sh
python3 confidence_gating.py --input labeling_results/<run>/labeling_results.jsonl
```

These examples measure behavior on synthetic cases. Model routing measures agreement with a routing policy; tool-call and inbox examples simulate decisions; game control uses fixed snapshots. Confidence thresholds are exploratory.

## Routing illustration

![Invoice extraction routing example](Invoice%20extraction%20routing%20example.png)
