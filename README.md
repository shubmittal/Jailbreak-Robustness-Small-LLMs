# Small Models, Same Rules

A reproducible, laptop-runnable **joint (ASR, FRR) audit** of four sub-4B open instruction-tuned LLMs under a single defensive system prompt.

> Companion artifact for the paper *Small Models, Same Rules: A Joint Attack-Success and False-Refusal Audit of Sub-4B Open LLMs Under a Single Defensive System Prompt* (`07_paper.md`). Audit-not-capability: **no new attack and no new defense** is proposed; all prompts come from previously published public benchmarks; no harmful completion is reproduced anywhere; only aggregate statistics and judge labels are released.

## What it measures

For each of **Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct, Qwen2.5-3B-Instruct, Gemma-2-2B-it**, with and without a ~60-word defensive system prompt:

- **ASR** (Attack Success Rate) on HarmBench-200 (+ optional GCG suffixes, JBB-Behaviors).
- **FRR** (False Refusal Rate) on XSTest-250 (+ optional OR-Bench-Hard).
- Triangulated across **three independent judges**: HarmBench fine-tuned classifier (Llama-2-13B, or Mistral-7B fallback), Llama-Guard-3-1B, and a deliberately weak keyword baseline. No judge falls back to another inside its own verdict.

## Repository map

| File | Purpose |
| --- | --- |
| `05_experiment.py` | Evaluation harness: loads models/judges/benchmarks at pinned revisions, runs the grid, writes `results.csv` + `aggregate.csv`. |
| `06_analysis.py` | Post-hoc analysis: Wilson CIs, paired bootstrap + McNemar, Pareto frontier, Kendall's τ ranking stability (all judge pairs), figures. |
| `defensive_prompts/` | The three pre-registered prompts (`primary`, `terse`, `constitutional`). SHA-256-checked at run start. |
| `08_preregistration.md` | Pre-registration (hypotheses, frozen revisions, analysis plan); public Git tag `prereg-v1`. Lock **before** running. |
| `07_paper.md` | Paper draft. | `references.bib` | Bibliography. |
| `tests/` | Schema smoke test linking the harness output to the analysis input. |
## Hardware

- **Primary:** one consumer NVIDIA GPU, 12–16 GB VRAM (RTX 4070-class). 4-bit quantization (bitsandbytes) is **CUDA-only**.
- **CPU-only fallback:** `--backend ollama`. On this path the HarmBench Llama-2-13B classifier is replaced by the Mistral-7B classifier or omitted (two-judge results).
- Full grid ≈ 12 GPU-hours (primary) / ≈ 18 GPU-hours (with replicates + ablations) — runnable overnight.

## Install

Python 3.10–3.12 with the canonical pins (used to produce the paper's numbers):

```bash
uv pip install -r requirements.txt      # or: pip install -r requirements.txt
```

Python 3.13: use the loosened bounds (`torch>=2.6`, `numpy>=2.1`, `scipy>=1.14`, `bitsandbytes>=0.45`):

```bash
uv pip install -r requirements-loose.txt
```

## Gated-model authentication

`meta-llama/Llama-3.2-3B-Instruct`, `google/gemma-2-2b-it`, and `meta-llama/Llama-Guard-3-1B` are gated. Accept each license on its Hugging Face page, then:

```bash
huggingface-cli login
```

The harness detects 401/403 errors and prints the exact license URLs to visit.

## Reproduce the paper

```bash
# 1. Verify the defensive prompt matches the pre-registered hash
python 05_experiment.py --check-prompt-hash <sha256-from-08_preregistration> --defense primary

# 2. Primary grid (keyword + Llama-Guard judges) — ~6 GPU-hours
python 05_experiment.py --backend transformers --n 200 \
  --judges keyword,llamaguard --defense primary --output_dir ./results/primary_grid

# 3. HarmBench classifier judge (sequenced separately in GPU memory) — ~3 GPU-hours
python 05_experiment.py --n 200 --judges harmbench --harmbench-cls-size large \
  --defense primary --output_dir ./results/harmbench_judge

# 4. Secondary benchmarks (JBB + OR-Bench-Hard)
python 05_experiment.py --enable-jbb --enable-orbench \
  --judges keyword,llamaguard --defense primary --output_dir ./results/secondary

# 5. GCG-suffix transfer (suffixes loaded from a public llm-attacks snapshot you provide)
python 05_experiment.py --gcg-suffix-file ./artifacts/gcg_suffixes.txt \
  --defense primary --output_dir ./results/gcg

# 6. Variant prompts + temperature robustness
python 05_experiment.py --defense terse          --output_dir ./results/terse
python 05_experiment.py --defense constitutional  --output_dir ./results/constitutional
python 05_experiment.py --temperature 0.7 --best-of-k 5 --output_dir ./results/t07

# 7. Analysis (Wilson CIs, bootstrap B=1000, Pareto, Kendall's tau)
python 06_analysis.py --results-csv ./results/primary_grid/results.csv \
  --out-dir ./results/primary_grid --bootstrap 1000 --ci 0.95
```

Outputs: `results.csv`, `aggregate.csv`, `run_manifest.json` (resolved revisions + prompt SHA), and the analysis tables/figures under `results/`.

## Data handling & ethics

- **No raw completions are written to disk by default.** Only per-prompt per-judge verdicts and aggregates.
- `--store-completions <path>` is opt-in and writes **Fernet-encrypted** records only; it refuses to write plaintext and requires `FERNET_KEY`.
- All benchmarks are used under their published licenses (HarmBench, JBB-Behaviors MIT; XSTest, OR-Bench CC-BY-4.0; AdvBench/GCG MIT). See paper §8 and Appendix E.

## Tests

```bash
python tests/test_pipeline_smoke.py     # or: pytest tests/
```
