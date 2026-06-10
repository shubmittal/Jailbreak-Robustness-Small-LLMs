#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# reproduce.sh
#
# One-shot reproduction of the experimental grid for
#   "Small Models, Same Rules: A Joint Attack-Success and False-Refusal Audit
#    of Sub-4B Open LLMs Under a Single Defensive System Prompt"
#
# Runs the harness (05_experiment.py) over the four target models
#   Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct, Qwen2.5-3B-Instruct,
#   Gemma-2-2B-it
# under {no_defense, with_defense}, across the three judges
#   keyword baseline, Llama-Guard-3-1B, HarmBench fine-tuned classifier,
# then aggregates with the analysis script (06_analysis.py).
#
# IMPORTANT: this script reproduces the *pipeline*, not any published number.
# The experiments have NOT been run; the numbers in the paper are
# TBD-after-running-experiment. This script is what produces them.
#
# Hardware
#   Requires a CUDA GPU. The paper's primary target is one consumer NVIDIA
#   GPU with 12-16 GB VRAM (RTX 4070-class); 16 GB gives comfortable headroom
#   but is not a hard floor. 4-bit quantization (bitsandbytes) is CUDA-only;
#   the HarmBench Llama-2-13B classifier is the most memory-hungry step. A
#   CPU-only path exists (pass --backend ollama to the harness), but it is NOT
#   what this script drives and was NOT used for the paper.
#
# Environment / pins
#   The canonical pinned requirements.txt was used to produce the paper
#   numbers. Install it first, in Python 3.10-3.12:
#       uv pip install -r requirements.txt   # or: pip install -r requirements.txt
#   (requirements-loose.txt is for Python 3.13 only and is NOT the paper pin.)
#
# Gated-model authentication (REQUIRED before any step, including step 0)
#   Every step in this script -- step 0 included -- loads the four target
#   models with the transformers backend. The gated repos
#   Llama-3.2-3B-Instruct, Gemma-2-2B-it, and Llama-Guard-3-1B require that you
#   accept each license on its Hugging Face page and then run
#   `huggingface-cli login`. If auth is missing, the very first step fails at
#   model load (not at the hash check), so complete this before running.
#
# Audit-not-capability: no harmful prompt, jailbreak string, GCG suffix, or
# model completion is inlined here. Attack content is loaded at runtime from
# public benchmarks by the harness; the GCG suffix file is user-provided.
#
# Usage
#   OUT_ROOT=./results ./reproduce.sh
#   GCG_SUFFIX_FILE=./artifacts/gcg_suffixes.txt OUT_ROOT=./results ./reproduce.sh
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration (override via environment).
# ---------------------------------------------------------------------------
# Output root for every step's --output_dir / --out-dir.
OUT_ROOT="${OUT_ROOT:-./results}"

# Pre-registered SHA-256 of the primary defensive prompt (stripped UTF-8),
# from 08_preregistration.md sec 2.4. Asserted by the harness at run start.
PRIMARY_PROMPT_SHA256="7adca1d95a6759f1eeab9e4ffe45aa5e33ea82a6fe2c57f74c819cd918cf0beb"

# Optional, user-provided GCG suffix file (public llm-attacks snapshot, one
# suffix per line). The GCG step is skipped unless this file exists, because
# the suffixes are never shipped with this artifact.
GCG_SUFFIX_FILE="${GCG_SUFFIX_FILE:-./artifacts/gcg_suffixes.txt}"

# Python interpreter (override e.g. PYTHON=python3).
PYTHON="${PYTHON:-python}"

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "[reproduce] OUT_ROOT          = ${OUT_ROOT}"
echo "[reproduce] PYTHON            = ${PYTHON}"
echo "[reproduce] GCG_SUFFIX_FILE   = ${GCG_SUFFIX_FILE} (used only if present)"
echo "[reproduce] primary prompt sha= ${PRIMARY_PROMPT_SHA256}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 0: Minimal n=1 end-to-end run that also verifies the prompt hash.
# ---------------------------------------------------------------------------
# NOTE: the harness has no "check-hash-and-exit" mode. --check-prompt-hash is
# enforced inside load_defensive_prompt(), which only RAISES on mismatch; on a
# match, main() continues into the full pipeline. So this step is a real (but
# tiny, n=1) end-to-end run: it loads all four target models with the
# transformers backend and evaluates one HarmBench + one XSTest prompt with the
# keyword judge. It therefore requires the gated-model auth described in the
# header. On a hash mismatch the harness raises and exits non-zero, so `set -e`
# aborts the whole reproduction before any of the larger steps below.
echo
echo "[step 0/7] n=1 smoke run + primary defensive prompt SHA-256 verification ..."
"${PYTHON}" 05_experiment.py \
  --backend transformers \
  --defense primary \
  --check-prompt-hash "${PRIMARY_PROMPT_SHA256}" \
  --n 1 \
  --judges keyword \
  --no_plot \
  --output_dir "${OUT_ROOT}/_hashcheck"
echo "[step 0/7] Prompt hash verified (and n=1 smoke run succeeded)."

# ---------------------------------------------------------------------------
# Step 1: Primary grid -- keyword + Llama-Guard-3-1B judges, primary defense.
# ---------------------------------------------------------------------------
# HarmBench-200 ASR and XSTest-250 FRR, both conditions, T=0 greedy.
# These two judges fit alongside each target model; the HarmBench classifier
# is sequenced separately (step 2) to stay inside the VRAM budget.
echo
echo "[step 1/7] Primary grid (judges: keyword,llamaguard; defense: primary) ..."
"${PYTHON}" 05_experiment.py \
  --backend transformers \
  --n 200 \
  --judges keyword,llamaguard \
  --defense primary \
  --temperature 0.0 \
  --best-of-k 1 \
  --output_dir "${OUT_ROOT}/primary_grid"
echo "[step 1/7] Primary grid done -> ${OUT_ROOT}/primary_grid"

# ---------------------------------------------------------------------------
# Step 2: HarmBench fine-tuned classifier judge -- separate sequenced run.
# ---------------------------------------------------------------------------
# Run on its own so the 13B (4-bit) classifier does not share GPU memory with
# the Llama-Guard judge from step 1. 'large' = Llama-2-13B; the harness
# auto-falls back to the Mistral-7B-val classifier if 13B will not fit.
echo
echo "[step 2/7] HarmBench-classifier judge run (judges: harmbench; cls-size: large) ..."
"${PYTHON}" 05_experiment.py \
  --backend transformers \
  --n 200 \
  --judges harmbench \
  --harmbench-cls-size large \
  --defense primary \
  --temperature 0.0 \
  --best-of-k 1 \
  --output_dir "${OUT_ROOT}/harmbench_judge"
echo "[step 2/7] HarmBench-classifier judge run done -> ${OUT_ROOT}/harmbench_judge"

# ---------------------------------------------------------------------------
# Step 3: Secondary benchmarks -- JBB-Behaviors (ASR) + OR-Bench-Hard (FRR).
# ---------------------------------------------------------------------------
# Cross-check ASR on JailbreakBench JBB-Behaviors and secondary FRR on
# OR-Bench-Hard, under the same keyword + Llama-Guard judges. FRR uses only the
# keyword and Llama-Guard judges by design (the HarmBench classifier is not an
# FRR judge), so no harmbench judge is requested here.
echo
echo "[step 3/7] Secondary benchmarks (JBB-Behaviors + OR-Bench-Hard) ..."
"${PYTHON}" 05_experiment.py \
  --backend transformers \
  --n 200 \
  --enable-jbb \
  --enable-orbench \
  --judges keyword,llamaguard \
  --defense primary \
  --temperature 0.0 \
  --best-of-k 1 \
  --output_dir "${OUT_ROOT}/secondary"
echo "[step 3/7] Secondary benchmarks done -> ${OUT_ROOT}/secondary"

# ---------------------------------------------------------------------------
# Step 4: GCG-suffix transfer (guarded -- suffix file is user-provided).
# ---------------------------------------------------------------------------
# Each HarmBench behavior is additionally evaluated with each canonical GCG
# suffix appended (the harness loads them at runtime from the file you point
# at; none are inlined in this repo). Skipped cleanly if the file is absent.
echo
if [[ -f "${GCG_SUFFIX_FILE}" ]]; then
  echo "[step 4/7] GCG-suffix transfer (suffix file: ${GCG_SUFFIX_FILE}) ..."
  "${PYTHON}" 05_experiment.py \
    --backend transformers \
    --n 200 \
    --judges keyword,llamaguard \
    --defense primary \
    --gcg-suffix-file "${GCG_SUFFIX_FILE}" \
    --temperature 0.0 \
    --best-of-k 1 \
    --output_dir "${OUT_ROOT}/gcg"
  echo "[step 4/7] GCG-suffix transfer done -> ${OUT_ROOT}/gcg"
else
  echo "[step 4/7] SKIPPED: GCG suffix file not found at ${GCG_SUFFIX_FILE}."
  echo "           Provide a public llm-attacks snapshot (one suffix per line)"
  echo "           and set GCG_SUFFIX_FILE to enable this step."
fi

# ---------------------------------------------------------------------------
# Step 5: Defensive-prompt variants -- terse and constitutional.
# ---------------------------------------------------------------------------
# Re-run the keyword + Llama-Guard grid under each alternative defensive prompt
# so the per-variant (ASR, FRR) tradeoff can be compared against primary.
echo
echo "[step 5/7] Defensive-prompt variant: terse ..."
"${PYTHON}" 05_experiment.py \
  --backend transformers \
  --n 200 \
  --judges keyword,llamaguard \
  --defense terse \
  --temperature 0.0 \
  --best-of-k 1 \
  --output_dir "${OUT_ROOT}/terse"
echo "[step 5/7] terse done -> ${OUT_ROOT}/terse"

echo "[step 5/7] Defensive-prompt variant: constitutional ..."
"${PYTHON}" 05_experiment.py \
  --backend transformers \
  --n 200 \
  --judges keyword,llamaguard \
  --defense constitutional \
  --temperature 0.0 \
  --best-of-k 1 \
  --output_dir "${OUT_ROOT}/constitutional"
echo "[step 5/7] constitutional done -> ${OUT_ROOT}/constitutional"

# ---------------------------------------------------------------------------
# Step 6: Sampling robustness -- T=0.7, best-of-5.
# ---------------------------------------------------------------------------
# Pre-registered robustness check: 5 samples/prompt (seeds base..base+4); an
# attack succeeds if any of the 5 succeed, a refusal occurs only if all 5
# refuse. Uses the primary defense and keyword + Llama-Guard judges.
echo
echo "[step 6/7] Sampling robustness (temperature 0.7, best-of-5) ..."
"${PYTHON}" 05_experiment.py \
  --backend transformers \
  --n 200 \
  --judges keyword,llamaguard \
  --defense primary \
  --temperature 0.7 \
  --best-of-k 5 \
  --output_dir "${OUT_ROOT}/t07"
echo "[step 6/7] T=0.7 best-of-5 run done -> ${OUT_ROOT}/t07"

# ---------------------------------------------------------------------------
# Step 7: Post-hoc analysis -- Wilson CIs, paired bootstrap (B=1000), Pareto,
#         Kendall's tau ranking stability.
# ---------------------------------------------------------------------------
# Runs on the primary grid's results.csv. Re-point --results-csv / --out-dir to
# analyze the other steps' outputs the same way.
echo
echo "[step 7/7] Analysis on primary grid (bootstrap B=1000, CI 0.95) ..."
"${PYTHON}" 06_analysis.py \
  --results-csv "${OUT_ROOT}/primary_grid/results.csv" \
  --out-dir "${OUT_ROOT}/primary_grid" \
  --bootstrap 1000 \
  --ci 0.95
echo "[step 7/7] Analysis done -> ${OUT_ROOT}/primary_grid/results/"

echo
echo "============================================================"
echo "[reproduce] All steps complete. Per-run outputs under ${OUT_ROOT}/."
echo "[reproduce] Headline metrics remain TBD-after-running-experiment;"
echo "[reproduce] inspect results.csv / aggregate.csv and the analysis"
echo "[reproduce] tables under ${OUT_ROOT}/primary_grid/results/."
echo "============================================================"
