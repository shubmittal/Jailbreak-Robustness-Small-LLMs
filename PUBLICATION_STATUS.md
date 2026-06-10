# Publication status & route to submission

_Last updated: 2026-06-07. Target: arXiv preprint → ACM FAccT 2027 (IEEE TAI fallback)._

This is a measurement paper whose central methodological claim depends on a **pre-registration timestamped before data collection** (a Git tag plus an arXiv v1 of the protocol — not OSF). The experiments have **not** been run yet, so the single most important thing is to lock the pre-registration and the revision pins **before** the GPU run starts — not after.

## Critical path (in order)

1. ✅ **Pin Hugging Face revisions to commit SHAs.** Done — `pin_revisions.py` resolved all 11 repositories; the SHAs are pasted into `05_experiment.py` (`DEFAULT_MODELS`, judge constants, dataset specs) and the `08_preregistration.md` §2 tables. *(Still to pin separately: the GCG-suffix snapshot URL + SHA-256, which is not a Hugging Face repo.)*
2. **Freeze the pre-registration** (`08_preregistration.md`): commit and tag it in the public repo, and post the protocol as arXiv v1; record the tag name + commit SHA + arXiv ID into paper §5.7 and Appendix I. *(Blocks the "pre-registered" framing in the abstract/§1.)*
3. **Run the grid** (≈18 GPU-hours). **← compute-blocked; this is the only GPU-bound step.** Currently deferred while another experiment uses the GPU.
4. **Fill Section 6.** Replace every `TBD-after-running-experiment` cell with `06_analysis.py` output; write the result prose around the real numbers (~1,500 words; the draft is ~1.5k under the 11k-word / 14-page FAccT target, almost entirely because §6 is placeholdered).
5. **Write Appendices C–I** (see below).
6. **Responsible disclosure** window: share the draft with the four model providers + four benchmark/judge maintainers 14–30 days before public release; log dates in Appendix I.

## Done this session (text/file work, no GPU)

- **Consistency fixes:** `max_new_tokens` 512→256 in the outline (now matches paper + code); Figure 5 judge "Llama-Guard-3-**8B**"→"**1B**"; "full grid 12h" vs "18h" framing reconciled to "primary 12h (~18h incl. ablations)" across abstract/outline/paper; "~450 prompts" now defined as HarmBench-200 + XSTest-250 wherever it appears.
- **Code↔prose alignment:** `06_analysis.py` now computes Kendall's τ for **all three judge pairs** (was only the first two — it previously under-delivered the three-judge claim behind H1); bootstrap default 2000→1000 to match the pre-registered B=1000, and the Appendix B reproduction command updated to match.
- **Citations:** added `gebru2021datasheets` and `mitchell2019modelcards` (named in §8.8 without a key) and cited them; cited the previously-orphaned `touvron2023llama2`; synced the `references.bib` header title. Paper↔bib now has **no missing keys and no orphans**.
- **Findings-vs-hypotheses language:** softened the three Discussion/Conclusion passages (§7.2 "Qwen2.5-3B already decodes simple Caesar", §7.4 "this finding", §10 "three durable implications emerge") that read as established results into hypothesis-conditioned framing. §6 itself was already clean (all `TBD`).
- **New deliverables:** `08_preregistration.md` (with the real SHA-256 hashes of the three defensive prompts), `README.md` (the "single-command reproduction" deliverable), and this status file.
- **Requirements:** `requirements-loose.txt` loosened for Python 3.13 (`torch>=2.6`, `numpy>=2.1`, `scipy>=1.14`, `bitsandbytes>=0.45`).
- Verified `06_analysis.py` imports, the smoke test passes, and the 3-judge τ structure produces all pairs.

## Appendices still to write (referenced in-text, not yet present)

The paper body forward-references Appendices C–I; only A and B exist. Each is a contained writing task:

- **C** — per-cell generation-error rates (needs run data).
- **D** — one-page practitioner checklist (writable now; named as a headline deliverable in §7.1 and §10).
- **E** — license-compliance checklist + data card (Gebru datasheet structure) + model card (Mitchell). Writable now.
- **F** — why the HarmBench classifier is not used as an FRR judge. Writable now.
- **G** — per-model chat-template versioning notes / pitfalls. Writable now.
- **H** — revision-drift 50-prompt check (needs run data).
- **I** — disclosure contact log + pre-registration tag / arXiv ID (needs steps 2 and 6).

## Open decisions for the author

- **HF revision SHAs** — which checkpoint date to pin (affects drift-check baseline in Appendix H).
- **Pre-registration mechanism** — RESOLVED (2026-06-09): skip OSF; use a Git tag + an arXiv v1 of the protocol for an independent, author-uncontrolled timestamp.
- **13B vs 7B HarmBench classifier** — confirm the 13B fits at 4-bit on the target GPU; if not, the run uses the Mistral-7B fallback and that choice is recorded in Appendix A.
- **GCG-suffix source** — pin the exact public `llm-attacks` snapshot URL + SHA for the 10 canonical suffixes.

## Known small items (non-blocking)

- Paper is ~1.5k words under the outlined 11k target — closes once §6 prose is written around real numbers.
- Appendix B per-step GPU-hour estimates (6h + 3h + unestimated secondary steps) don't visibly sum to the 12h primary-grid line; cosmetic, optional to itemize.
- A `reproduce.sh` and `environment.yml` are referenced (Appendix A) but not yet in the repo; add them so the "single shell command" claim is literally true.
