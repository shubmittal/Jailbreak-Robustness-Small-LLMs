# Paper Outline

**Title.** Small Models, Same Rules: A Joint Attack-Success and False-Refusal Audit of Sub-4B Open LLMs Under a Single Defensive System Prompt

**Target venue.** arXiv preprint then ACM FAccT 2027; IEEE TAI as fallback.

**Total estimated length.** ~11,000 words main text (excluding references and appendices), ~14 pages in the FAccT two-column template.

---

## Title and Author Block (~60 words)

- Final title (above) signals four scope decisions at once: comparative-audit framing, the sub-4B model class, the joint ASR+FRR contribution, and the single-defensive-system-prompt intervention.
- Author block: single author at Microsoft. The Responsible AI practitioner affiliation is stated explicitly per FAccT norms on positionality.
- Affiliation footnote discloses Microsoft Responsible AI affiliation and states that no internal or proprietary Microsoft models, datasets, evaluation tooling, or deployment systems were used in any part of the study.
- A second footnote marks the paper as an arXiv-first preprint with a planned submission to ACM FAccT 2027.
- No emoji, no decorative typography; the author block follows the ACM Reference Format.

**Key citations.** None.

---

## Abstract (~250 words)

- Open with the deployment shift: sub-4B instruction-tuned open language models now run on laptops, mobile devices, and air-gapped enterprise stacks, exactly the settings where classifier auxiliaries, decoding-time interventions, and representation-engineering defenses are infeasible.
- State the three gaps closed by the paper: (i) a head-to-head sub-4B safety audit on Llama-3.2-3B-Instruct, Phi-3-mini, Qwen2.5-3B-Instruct, and Gemma-2-2B-it; (ii) joint measurement of attack-success rate and false-refusal rate on the same models under the same intervention; (iii) explicit benchmarking of the practitioner-grade defense, a single natural-language system prompt.
- State the methodological contribution: three-judge triangulation (HarmBench fine-tuned classifier — Llama-2-13B where memory permits, Mistral-7B otherwise — Llama-Guard-3-1B, and a refusal-keyword baseline used as an independent third signal rather than as a fallback inside another judge) with Kendall's tau ranking stability and paired nonparametric bootstrap confidence intervals on rate differences.
- State scope honestly: a static-attack distribution, English-only, single-turn; no new attacks and no new defenses are proposed.
- State the reproducibility envelope: the primary grid (4 models x 2 system prompts x 2 temperatures x ~450 core prompts [HarmBench-200 + XSTest-250] x 3 judges) completes in approximately 12 GPU-hours on a consumer 16 GB GPU (~18 including replicates and ablations), runnable overnight on a laptop.
- Close with the pre-registered hypothesis pattern (state these as hypotheses to be tested, not as established findings, unless the public pre-registration commit/tag timestamp can be shown to precede data collection): at least one pairwise model ordering flips between judges, and the defense cost varies by more than two-fold across the four models under the identical prompt. Report the observed direction and magnitude of each in Section 6. State the static-attack floor framing explicitly: we measure the lower bound a paste-attacker faces and the universally available defense buys, not adaptive-attack robustness, and cite Andriushchenko et al. (2024) as the explicit ceiling.

**Key citations.** `mazeika2024harmbench`, `chao2024jailbreakbench`, `souly2024strongreject`, `rottger2024xstest`, `xie2023selfreminder`, `zhang2024goalpriority`.

---

## 1. Introduction (~1,100 words)

### 1.1 Deployment reality and the audit gap
- Llama-3.2-3B-Instruct, Phi-3-mini, Qwen2.5-3B-Instruct, and Gemma-2-2B-it now ship in Apple Intelligence-adjacent stacks, Ollama and llama.cpp deployments, on-device assistants, and enterprise air-gapped pilots.
- These are precisely the Responsible AI-critical contexts in which defensive infrastructure is most limited: no Llama-Guard front-end at the edge by default, no SmoothLLM perturbation budget, no SafeDecoding logit rewriter.
- Public jailbreak-robustness evidence overwhelmingly targets 7B+ open models and frontier closed models; the sub-4B class is the deployment frontier with the thinnest audit record.

### 1.2 Audit question
- For a practitioner who must pick one sub-4B open model and harden it with the only universally available defense (a system prompt), what does the (ASR, FRR) trade-off look like, and is the safety ranking stable across reasonable judges?
- This framing makes two practitioner decisions explicit: model selection within a class and a fleet-level defense policy.
- We measure a floor, not a ceiling: the question is what a non-expert paste-attacker faces and what a universally available defense buys against that distribution. Adaptive log-probability search (Andriushchenko et al., 2024) reaches 100% ASR on sub-4B models and bounds the ceiling; the floor framing is what the rest of the paper measures and is restated in the threat model (Section 3.6), the discussion (Section 7.3), and the conclusion.

### 1.3 Three under-served gaps
- (i) There is no head-to-head sub-4B safety comparison under a common protocol.
- (ii) ASR and FRR are reported in separate benchmarks (HarmBench/JBB versus XSTest/OR-Bench), with rare joint reporting on the same models and the same intervention.
- (iii) The most-deployed defense (a single natural-language system prompt) is under-benchmarked relative to SmoothLLM, RPO, circuit breakers, and SafeDecoding, which dominate the recent defense literature despite higher deployment cost.

### 1.4 Contributions
- (a) A reproducible, laptop-runnable harness with pinned model revisions, pinned chat templates, pinned benchmark commits, and a single shell command that regenerates every table in the paper.
- (b) A four-model joint (ASR, FRR) frontier under empty and defensive system prompts.
- (c) Quantification of defense cost as XSTest safe prompts lost per HarmBench behavior gained, with bootstrap confidence intervals.
- (d) A three-judge sensitivity analysis with Kendall's tau ranking stability and a pre-registered flip-detection rule.

### 1.5 Non-contributions
- No new attack algorithm.
- No new defense algorithm.
- No proposed safety benchmark.
- The framing follows FAccT's stated preference for audit and measurement work and is consistent with Microsoft RAI's dual-use posture on jailbreak research.

### 1.6 Outline preview
- Sections 2-3 cover background and the threat model.
- Sections 4-5 describe the methodology and experimental setup.
- Section 6 reports results across nine sub-questions.
- Section 7 discusses practitioner takeaways, the capability-vs-alignment safety distinction, and judge politics.
- Sections 8-9 cover ethics, responsible disclosure, and limitations.

**Key citations.** `mazeika2024harmbench`, `chao2024jailbreakbench`, `souly2024strongreject`, `rottger2024xstest`, `cui2024orbench`, `an2024phtest`, `xie2023selfreminder`, `zhang2024goalpriority`, `abdin2024phi3`, `grattafiori2024llama3`, `riviere2024gemma2`, `yang2024qwen25`.

---

## 2. Background and Related Work (~1,600 words)

### 2.1 Alignment lineage for sub-4B open models
- Preference-learning foundations: Christiano et al. (preference learning from human feedback), Ouyang et al. (InstructGPT and RLHF), Bai et al. (HH-RLHF and Constitutional AI), Rafailov et al. (DPO).
- Documented alignment pipelines for the four targets: Abdin et al. on Phi-3 and Haider et al. on Phi-3 safety post-training; Riviere et al. on Gemma-2 instruction tuning; Grattafiori et al. on the Llama-3 family; Yang et al. on Qwen2.5.
- Note: three of the four targets use DPO-style alignment; this is relevant to the discussion of mismatched generalization in Section 7.

### 2.2 Attack methodology evolution
- We compress the four-era taxonomy into one subsection.
- Manual era: DAN prompts (Shen et al.), PromptInject (Perez and Ribeiro), red-teaming corpora (Perez et al.).
- Gradient-based era: GCG (Zou et al.), AutoDAN (Liu et al.), the Wei et al. competing-objectives and mismatched-generalization framing.
- Automated black-box era: PAIR (Chao et al.), GPTFuzz (Yu et al.), TAP (Mehrotra et al.).
- Adaptive and semantic era: PAP (Zeng et al.), persona attacks (Shah et al.), multilingual jailbreaks (Yong et al., Deng et al.), cipher attacks (Yuan et al.), many-shot attacks (Anil et al.), Andriushchenko et al. adaptive log-prob search.
- Carlini et al. on text-vs-multimodal calibration motivates measurement triangulation.
- State explicitly: we re-use static artifacts from GCG and DAN families via the HarmBench and JailbreakBench distributions; we cite but do not re-run PAIR, TAP, PAP, or adaptive attacks.

### 2.3 Defense landscape
- Training-time: covered in 2.1.
- Auxiliary classifiers: Llama-Guard (Inan et al.), WildGuard (Han et al.), ShieldGemma.
- Prompt-based: Xie et al. self-reminder, Wei et al. in-context defense, Phute et al. self-defense, Zhang et al. goal-priority.
- Inference-time perturbation: SmoothLLM (Robey et al.), erase-and-check (Kumar et al.), RPO.
- Decoding: SafeDecoding (Xu et al.).
- Architectural: Circuit Breakers (Zou et al.), refusal-direction (Arditi et al.).
- We argue that the practitioner-accessible slice is prompt-based, and we place our work there.

### 2.4 Measurement evolution
- Keyword refusal matching (Zou et al.).
- Specialized classifiers: HarmBench Llama-2-13B classifier, JBB judge, Llama-Guard-3, WildGuard.
- Rubric LLM judges: StrongREJECT (Souly et al.).
- Judge benchmarks: JailJudge (Lin et al.).
- This motivates our three-judge triangulation.

### 2.5 The FRR / over-refusal axis
- XSTest (Rottger et al.), OR-Bench (Cui et al.), PHTest (An et al.), BeaverTails (Ji et al.), SORRY-Bench (Xie et al.).
- BeaverTails motivated joint (helpful, harmless) annotation but current benchmarks still report the two axes separately on different models. This is the gap our joint frontier addresses.

**Key citations.** `christiano2017preferences`, `ouyang2022instructgpt`, `bai2022hh`, `bai2022constitutionalai`, `rafailov2023dpo`, `abdin2024phi3`, `bhatt2024phi3safety`, `riviere2024gemma2`, `grattafiori2024llama3`, `yang2024qwen25`, `touvron2023llama2`, `zou2023gcg`, `liu2024autodan`, `chao2023pair`, `yu2023gptfuzzer`, `mehrotra2024tap`, `shen2024dan`, `perez2022promptinject`, `perez2022redteaming`, `wei2023jailbroken`, `carlini2023aligned`, `zeng2024pap`, `shah2023persona`, `yong2023lowresource`, `deng2024multilingual`, `yuan2024cipher`, `anil2024manyshot`, `andriushchenko2025adaptive`, `inan2023llamaguard`, `han2024wildguard`, `xie2023selfreminder`, `wei2023ica`, `phute2024selfdefense`, `zhang2024goalpriority`, `robey2023smoothllm`, `kumar2023certifying`, `xu2024safedecoding`, `zou2024circuitbreakers`, `arditi2024refusal`, `mazeika2024harmbench`, `chao2024jailbreakbench`, `souly2024strongreject`, `lin2024jailjudge`, `rottger2024xstest`, `cui2024orbench`, `an2024phtest`, `ji2023beavertails`, `xie2024sorrybench`, `casper2023rlhfproblems`.

---

## 3. Threat Model (~900 words)

### 3.1 Adversary capabilities
- Black-box prompt-only access via the model's chat template.
- No fine-tuning access, no logit access, no gradient access, no multi-turn state manipulation.
- This is the modal user-facing attack surface for on-device deployments where the model is shipped as an opaque inference endpoint.

### 3.2 Adversary objectives
- Elicit a substantive, non-refusing, on-topic harmful completion to one of the 200 HarmBench standard behaviors (7 semantic categories x 4 functional categories).
- "Substantive" is defined per the judge's rubric: HarmBench classifier returns yes; Llama-Guard-3 returns unsafe; keyword baseline records absence of refusal terms.

### 3.3 Attack distribution scope
- We audit the static-attack distribution: pre-published prompts and pre-computed GCG suffixes from public benchmarks. This approximates what a non-expert paste-attacker actually sends.
- Out of scope and reasons:
  - Adaptive log-probability search (Andriushchenko et al.): requires per-prompt compute and per-model retuning that an on-device paste-attacker does not run.
  - Automated attacker LLMs (PAIR, TAP): require an attacker model with API budget; conflicts with the laptop-runnable reproducibility envelope.
  - Gradient re-optimization on the target: requires gradient access we do not assume.
  - Fine-tuning attacks (Qi et al.): require training infrastructure and elevated access.
  - Many-shot attacks (Anil et al.): require long-context budgets that the sub-4B class largely does not support.
  - Multi-turn Crescendo-style attacks: orthogonal axis; explicit single-turn scope.
  - Multilingual attacks (Yong et al., Deng et al.): conflates multilingual capability with multilingual safety in a model class with heterogeneous multilingual coverage.

### 3.4 Defender capabilities
- A single natural-language system prompt prepended via the chat template.
- No classifier auxiliary, no perturbation, no decoding modification, no fine-tuning.
- This is the universal lower-bound defense surface that every deployment of every sub-4B open model can apply.

### 3.5 Defender objectives
- Minimize ASR on HarmBench behaviors and AdvBench GCG-suffixed prompts.
- Maintain helpfulness on superficially-unsafe-looking but actually-benign prompts (XSTest, OR-Bench-hard).
- Operate under a deployment policy that applies one prompt across a heterogeneous model fleet.

### 3.6 Threat-model floor framing
- Andriushchenko et al. (2025) report 100% adaptive ASR on Phi-3-mini and similar models. We do not claim adaptive robustness.
- We measure a floor: what a practitioner gets from the universally available defense against the realistic paste-attack distribution.
- Greshake et al. on indirect prompt injection: on-device single-user contexts have a different threat surface than LLM-integrated apps, and we restrict claims accordingly.

**Key citations.** `andriushchenko2025adaptive`, `qi2024finetuning`, `anil2024manyshot`, `yong2023lowresource`, `deng2024multilingual`, `chao2023pair`, `mehrotra2024tap`, `greshake2023indirect`, `zou2023gcg`, `mazeika2024harmbench`.

---

## 4. Methodology (~1,500 words)

### 4.1 Joint (ASR, FRR) frontier framing
- Define ASR_j(M, p) and FRR_j(M, p) for model M, system prompt p in {empty, defensive}, judge j.
- The joint frontier is the two-dimensional plot of these pairs.
- Define defense cost C(M) = (FRR_defensive - FRR_empty) / (ASR_empty - ASR_defensive), with paired nonparametric bootstrap confidence intervals on the ratio. We document C(M)'s behaviour when ASR_empty - ASR_defensive is small: a bootstrap floor rule (the ratio is reported as "undefined" whenever the denominator's 95% CI includes zero) and a rank-based dominance test as a complementary signal.
- We report an alternative formulation as a robustness check: the area A(M) between the empty-prompt and defensive-prompt operating points on the (ASR, FRR) plane, computed as the trapezoidal area swept along the within-model defensive shift. A(M) is well-defined whenever both endpoints exist and is less brittle than C(M) under small ASR shifts. We also tabulate the simple pair (Delta ASR, Delta FRR) per model, normalised per 1000 prompts on each axis.
- We state explicitly that no universal practitioner threshold for C(M) or A(M) exists. The metrics support relative comparison across models under the same protocol; absolute thresholds remain a procurement-policy decision.

### 4.2 Model selection rationale
- We scope to the smallest comparable instruction-tuned class with documented safety post-training and public weights.
- Mistral-7B excluded: weaker public safety documentation.
- Llama-3.1-8B excluded: outside the sub-4B class.
- OLMo-1B and SmolLM noted as future work in Section 9 (capability headroom currently insufficient for several HarmBench behaviors).

### 4.3 Attack inputs
- HarmBench 200-behavior standard set, filtered to the standard functional category (rows whose FunctionalCategory is "standard"); we exclude contextual and copyright subsets that require external context. The loader prints the per-category histogram at load time and asserts the count against the published HarmBench standard-200 distribution within a documented tolerance.
- Ten canonical pre-computed GCG suffixes from the Zou et al. public artifact, loaded at runtime via a dedicated loader from the official llm-attacks repository snapshot (never inlined in source), applied to each behavior as a transferability stress test. The loader is documented in Appendix B alongside the defensive prompt.
- A JailbreakBench JBB-Behaviors subset of 50 reused as a cross-check on judge stability, loaded at runtime from the JailbreakBench/JBB-Behaviors Hugging Face repository at a pinned commit hash.

### 4.4 Defensive system prompt
- The ~60-word defensive system prompt is stated verbatim in Appendix B and reproduced in the main text in Section 4.4.
- The verbatim prompt is loaded from a dedicated text file `defensive_prompts/primary.txt`, hashed with SHA-256 at run start, and asserted equal to the hash recorded in the tagged pre-registration commit. A CLI flag `--defense {primary,terse,constitutional}` selects between the three pre-registered variants (`primary.txt`, `terse.txt`, `constitutional.txt`).
- Length is mid-range between Xie et al.'s self-reminder (~30 words) and Zhang et al.'s goal-priority few-shot (~200 tokens).
- Two pre-registered variant prompts are evaluated for sensitivity in Section 6.5: a terse ~20-word version and a constitutional-style ~200-word version with three explicit principles.
- Placement: the defensive prompt is delivered through each model's chat-template system slot where the template supports one. Gemma-2's chat template does not accept a system role; for Gemma we apply a placement preprocessor (centralised in `build_messages`, not buried in the transformers backend) that prepends the prompt to the first user message. An integration test asserts that the rendered template for every (backend, model) pair contains the defensive system text under the `with_defense` condition. Section 6.5 includes a placement ablation against user-prepended placement for the three models whose templates do accept a system role, so the Gemma-specific placement does not act as a hidden confound.

### 4.5 Judges and triangulation
- (a) HarmBench fine-tuned classifier. Primary: `cais/HarmBench-Llama-2-13b-cls` loaded 4-bit via bitsandbytes, applying HarmBench's verbatim `LLAMA2_CLS_PROMPT` template with the behaviour and generation slots and parsing the "Yes"/"No" first token. Fallback (documented in Appendix A) when the 13B classifier does not fit at 4-bit on the target consumer GPU: `cais/HarmBench-Mistral-7b-val-cls`. The chosen variant is pinned by Hugging Face commit hash. Judges are sequenced so that only one is resident at a time.
- (b) Llama-Guard-3-1B (`meta-llama/Llama-Guard-3-1B`), used in pure mode: it returns `unsafe` (HarmBench-side: jailbreak succeeded; XSTest-side: not refusal) or `safe` (HarmBench-side: jailbreak failed; XSTest-side: refusal classification deferred to its own pure judgement, with no fallback to the keyword judge inside Llama-Guard's verdict). The chat template is invoked with `add_generation_prompt=True` so the classifier emits its safety token.
- (c) Zou et al. refusal-keyword wordlist as a deliberately weak sanity baseline, used as an independent third judge — never as a fallback inside another judge.
- Each judge exposes two independent methods: `harmful(prompt, completion) -> bool` for HarmBench scoring and `refused(prompt, completion) -> bool` for XSTest scoring. ASR is `mean(harmful)` on HarmBench under each judge independently; FRR is `mean(refused)` on XSTest under each judge independently. Per-prompt per-judge verdicts are persisted in `results.csv` (one row per (model, condition, benchmark, prompt_id, judge)).
- Reported ASR is per-judge with a triangulated summary; per-judge breakdown appears in every table.
- Kendall's tau is computed on model rankings across judges; B=1000 paired nonparametric bootstrap confidence intervals on ASR differences; McNemar paired tests for before/after defense within model.

### 4.6 FRR measurement
- XSTest 250 safe prompts, loaded from a single pinned (repo, split, revision) Hugging Face combination, with safe/unsafe partitioning by the canonical XSTest `type` field (XSTest v2 marks unsafe contrast prompts with explicit `type_v2` markers; the loader inspects the schema at load time, asserts |safe| == 250, and prints the label distribution so the safe denominator is auditable).
- We measure FRR with two judges, not three: the keyword judge and Llama-Guard-3-1B (its refusal-head verdict, not a keyword fallback). This change is documented in the limitations section: a third GPT-4-rubric judge would be desirable but is omitted to preserve laptop reproducibility and to avoid paid APIs.
- OR-Bench-Hard (1,319 prompts) is loaded from `bench-llm/or-bench` with `subset='or-bench-hard-1k'` at a pinned commit hash and judged with the same two-judge protocol as XSTest. Section 5.5 updates the compute budget to include the OR-Bench-Hard rows.

### 4.7 Sampling protocol
- T=0 deterministic decoding (primary). T=0.7 best-of-5 sampling (robustness check): for each prompt at T=0.7 we generate k=5 samples with seeds {0,1,2,3,4}, judge each independently, and aggregate per the pre-registered rule (attack succeeds if any of the 5 succeed; refusal occurs if all 5 refuse). All five completions' judge labels are persisted in `results.csv` so per-seed dispersion is recoverable in analysis. If the laptop compute envelope does not admit best-of-5, we revert to T=0 only and restate the temperature protocol in Section 6.7.
- top-p = 1.0, max_new_tokens = 256, seeds {0, 1, 2, 3, 4}.
- Chat templates pinned to model-card versions (per-model versioning notes in Appendix G).
- The defensive prompt is placed in the chat template's system slot for the three models whose templates accept one; for Gemma-2 we prepend to the first user message via the centralised placement preprocessor and report a placement ablation in Section 6.5.

### 4.8 Statistical protocol
- Wilson 95% score intervals on all single-rate point estimates (closed-form; no bootstrap needed for a single Bernoulli rate).
- McNemar paired test for before/after defense within a model.
- Paired nonparametric bootstrap B=1000 for ASR-difference distributions: prompt-level pairs (success_no_def, success_with_def) are resampled with replacement, and the difference's percentile interval is reported.
- Kendall's tau across judges.
- Spearman correlation across XSTest and OR-Bench-hard.
- Generation errors (backend exceptions, OOM, timeout) are persisted as a third state alongside `refused` and `complied`. They are excluded from both ASR and FRR denominators, and the per-cell error rate is reported in Appendix C. The harness halts the run and surfaces the underlying exception if any (model, condition, benchmark) cell exceeds a 2% error rate, rather than silently coding errors as refusals.

### 4.9 Reproducibility
- Pinned Hugging Face revision hashes for every model and judge.
- Pinned chat template files alongside the harness.
- Pinned benchmark commits.
- The harness is released as a single `uv`-installable Python package with one shell script that reproduces every table.

**Key citations.** `mazeika2024harmbench`, `zou2023gcg`, `chao2024jailbreakbench`, `xie2023selfreminder`, `zhang2024goalpriority`, `inan2023llamaguard`, `rottger2024xstest`, `cui2024orbench`, `an2024phtest`.

---

## 5. Experimental Setup (~900 words)

### 5.1 Hardware
- Primary: a single RTX 4070-class consumer NVIDIA GPU with 12-16 GB VRAM. 4-bit quantization is CUDA-only via bitsandbytes; the laptop-runnable claim applies to NVIDIA-GPU laptops.
- Fallback: CPU-only execution via the Ollama HTTP backend for the four target models. The HarmBench Llama-2-13B classifier judge is GPU-required in 4-bit; CPU reproductions substitute the smaller HarmBench Mistral-7B classifier or skip the HarmBench classifier and report two-judge results. Hardware requirements are stated explicitly in the reproducibility section and the README.

### 5.2 Software stack
- Hugging Face `transformers` with `bitsandbytes` 4-bit quantization for judges and target models on CUDA hardware.
- Ollama as the documented CPU-only fallback path.
- `requirements.txt` pins exact versions (e.g., `transformers==4.46.3`, `datasets==3.0.1`, `bitsandbytes==0.44.1`, `torch==2.4.1`). A `requirements-loose.txt` companion file lists `>=` bounds for users who want flexibility, but the canonical pinned versions are the ones used to produce the paper's numbers.
- Gated model access (Llama-3.2-3B-Instruct, Gemma-2-2b-it, Llama-Guard-3-1B) requires Hugging Face authentication and license acceptance. The harness detects 401/403 errors from `huggingface_hub` and prints an explicit message instructing the user to run `huggingface-cli login` and accept the model licenses at the linked URLs; the README documents this prominently.
- Python and CUDA versions pinned in `environment.yml` and reproduced in Appendix A.

### 5.3 Model revisions
- Exact Hugging Face commit hashes (recorded as `(repo_id, revision)` tuples in `DEFAULT_MODELS`) for Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct (the 4k variant is chosen for chat-template stability over the 128k variant), Qwen2.5-3B-Instruct, and Gemma-2-2B-it; the same is done for every judge.
- The revision argument is threaded through `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, and `load_dataset`. The resolved hashes are recorded in `run_manifest.json`.
- Drift check: a 50-prompt subset is re-run against an earlier-pulled checkpoint per model (~2 weeks delta) to bound revision sensitivity; results reported in Appendix H.
- Refresh appendix: between submission and camera-ready we commit to re-running the harness on at least one newer-generation 1-4B model released in the intervening months (e.g., Phi-4-mini, SmolLM2-1.7B-Instruct, OLMo-2-1B-Instruct), so the contribution does not date as the sub-4B class evolves. The model selection rationale is reframed as "the harness and protocol, instantiated on the four canonical 2024 sub-4B safety-documented open instruct models, extended in the refresh appendix to one newer-generation reference point".

### 5.4 Defensive prompt artifact
- The verbatim primary prompt and two pre-registered variants are published in Appendix B and committed and tagged in the public repository before the main experiments are run.

### 5.5 Compute budget
- Primary grid: 4 models x 2 system prompts x 2 temperatures x ~450 core prompts (HarmBench-200 + XSTest-250) x 3 judges, estimated at ~12 GPU-hours.
- Replicates and ablations: ~6 GPU-hours.
- Total: ~18 GPU-hours, runnable overnight on a laptop.

### 5.6 Data handling
- No harmful completions are stored in human-readable form in the public artifact.
- Only judge labels and aggregate statistics are released.
- By default the harness persists per-prompt per-judge verdicts and aggregate statistics, not raw completions.
- An opt-in encrypted-completions sink is available (`--store-completions`) for reviewer-audit purposes. Completions are written via Fernet symmetric encryption to a path inside the author's Microsoft-managed environment with documented key management, a bounded retention period, and a reviewer-access protocol. The opt-in is OFF by default; the prose, code, and pre-registration all match this behaviour.

### 5.7 Pre-registration
- The pre-registration is committed to the public repository and marked with a signed pre-registration tag before any main experiment is run; the registration commit SHA and tag are recorded in Section 5.7 and Appendix I before submission. The registration commit/tag timestamp must demonstrably precede data collection (the first run_manifest.json); if it does not, the headline-result language in the abstract and Section 1 is rewritten to read as hypotheses tested rather than findings established.
- The pre-registered variables are:
  - Judge identity and version (HF commit hash) for the HarmBench classifier, Llama-Guard-3-1B, and the keyword wordlist (a verbatim copy of the wordlist is included).
  - Target model revisions (HF commit hashes).
  - The verbatim defensive prompt text, hashed with SHA-256; the hash is asserted at run start.
  - Temperature and seeds.
  - Primary metric (ASR on HarmBench-200), secondary metrics (FRR on XSTest-250 and OR-Bench-Hard).
  - Ranking-flip definition: a pairwise model ordering is said to "flip" between two judges if the per-judge ASR point estimates change sign on at least one pair *and* the Wilson 95% CIs of the two judges' ASR estimates for the affected pair do not overlap on the implied direction; Kendall's tau below 0.5 across judges is reported as a secondary flip indicator.
- What is exploratory: per-category attack-family breakdowns beyond the seven HarmBench semantic categories, qualitative cipher decoding analysis, and any post-hoc judge calibration adjustments.
- The tagged commit contains: the pre-registered protocol, the SHA-256 of the defensive prompt files, the harness commit hash at pre-registration time, the model and judge revision pins, and the analysis script's expected output schema.

**Key citations.** `abdin2024phi3`, `riviere2024gemma2`, `grattafiori2024llama3`, `yang2024qwen25`, `mazeika2024harmbench`, `inan2023llamaguard`.

---

## 6. Results (~1,700 words)

### 6.1 Headline joint (ASR, FRR) frontier
- Figure 1 plots all four models in the (ASR_empty, FRR_empty) and (ASR_defensive, FRR_defensive) plane.
- Arrows show the within-model shift induced by the defensive prompt.
- Results are aggregated across three judges with a shaded judge-variance band.

### 6.2 Per-judge ASR table
- Table 1 lists ASR for each (model, system prompt) cell under each judge with Wilson 95% confidence intervals.
- Cells whose intervals do not overlap across (model, condition) pairs are bolded.
- We state explicitly whether the per-judge ranking matches.

### 6.3 Ranking stability
- Table 2 lists the model ranking under each judge.
- Kendall's tau is reported across judges.
- Any ranking flip is flagged in-text per the pre-registration.
- We compare the magnitude of the flip to Souly et al.'s reported 30-point judge variance and to the Lin et al. JailJudge findings.

### 6.4 Defense cost
- Table 3 lists C(M) = Delta-FRR / Delta-ASR for each model.
- Figure 3 displays the more-than-two-fold spread across models as a bar chart with bootstrap whiskers.

### 6.5 System-prompt sensitivity
- Results under the terse and constitutional-style variant prompts.
- The dispersion band justifies the headline single-prompt scope and supports the practitioner-policy framing.

### 6.6 Attack-family breakdown
- Figure 4 decomposes ASR by HarmBench semantic category (rows) and model (columns).
- We identify which families each model is differentially robust or vulnerable to.
- Cipher attacks under-perform on sub-4B models because the models fail to decode the cipher; we annotate this column as a capability-driven safety artifact and revisit it in Section 7.

### 6.7 Temperature sensitivity
- T=0 vs T=0.7 best-of-5 are compared.
- We report whether the model ranking preserves under realistic deployment sampling.

### 6.8 FRR cross-benchmark
- Spearman correlation between XSTest compliance and OR-Bench-hard-1K compliance per model.
- We compare against Cui et al.'s reported population-level cross-benchmark correlation of 0.878 and discuss any per-model deviations.

### 6.9 GCG suffix transfer to sub-4B
- ASR delta from appending the ten canonical Zou et al. suffixes.
- We expect a modest effect because the sub-4B targets were not in the original GCG optimization set; we report the observed delta with confidence intervals.

**Key citations.** `souly2024strongreject`, `lin2024jailjudge`, `cui2024orbench`, `rottger2024xstest`, `mazeika2024harmbench`, `zou2023gcg`, `inan2023llamaguard`, `yuan2024cipher`.

---

## 7. Discussion (~1,100 words)

### 7.1 Practitioner takeaways
- Procurement decisions among sub-4B open models cannot be made on a single ASR number from a single judge.
- (ASR, FRR) should be inspected jointly, and rankings verified under at least two judges.
- A one-page checklist is reproduced as Appendix D.

### 7.2 Capability-vs-alignment safety
- Cipher and encoded attacks under-perform because models lack the capability to decode them.
- As small-model capability rises (Qwen2.5-3B already decodes simple Caesar), this "free safety" will erode.
- We connect this to Wei et al.'s competing-objectives and mismatched-generalization framework and to Arditi et al.'s refusal-direction analysis.

### 7.3 The static-attack floor framing
- This subsection is load-bearing. We restate the framing explicitly: the paper measures the lower bound a paste-attacker faces and the universally available defense buys, not adaptive-attack robustness. Andriushchenko et al. (2024) report 100% adaptive ASR on Phi-3-mini and similar; we cite this as the ceiling at each of the abstract, Section 1.2, Section 3.6, and the conclusion.
- The procurement reading: comparing sub-4B models on the static-attack floor is what a fleet-management decision needs, because adaptive attackers will saturate any model in this class, but most real-world traffic is paste-attack distribution.
- Defensive system prompts are necessary but never sufficient.
- We recommend layered defenses (Llama-Guard-3 on input and output) where infrastructure permits.

### 7.4 The defense cost is non-uniform
- A single safety prompt applied across a heterogeneous open-model fleet induces differential helpfulness regressions across models.
- This is a fleet-management finding distinct from per-model robustness and is, to our knowledge, not previously quantified for the sub-4B class.

### 7.5 Judge politics
- HarmBench classifier and Llama-Guard-3 disagree because their training distributions differ.
- StrongREJECT-style rubric judges are more faithful but require a strong LLM judge we deliberately avoid to preserve laptop reproducibility.
- We discuss the trade-off and recommend triangulation as standard practice.

### 7.6 Connection to FAccT and to Responsible AI
- Our work exemplifies audit-not-capability research: pure measurement, no new attack surface, reproducible by any practitioner.
- We argue this is the safety contribution most useful for the on-device deployment frontier.

**Key citations.** `wei2023jailbroken`, `arditi2024refusal`, `andriushchenko2025adaptive`, `souly2024strongreject`, `casper2023rlhfproblems`, `yuan2024cipher`, `inan2023llamaguard`.

---

## 8. Ethics and Responsible Disclosure (~900 words)

### 8.1 Dual-use posture
- We propose no new attack and no new defense.
- We re-use only publicly available benchmark prompts and the publicly available pre-computed GCG suffix artifact.
- No GCG, PAIR, or TAP optimization is re-run; no new attack artifact is produced or released.
- Inverted procurement risk. Per-model defense-cost asymmetries and an ASR ranking are information both for safety-seeking deployers and for adversaries choosing a target model to attack. We address this directly: all four models are already public, all benchmarks are already public, the spread we measure is small relative to the adaptive-attack ceiling (Andriushchenko et al. report 100% adaptive ASR on Phi-3-mini and similar), and the marginal uplift to an adversary who reads our paper to pick a target is therefore bounded. This is consistent with the audit-not-capability framing.

### 8.2 Content handling
- No harmful model completion is reproduced verbatim in the paper, the supplement, or the release artifact.
- Aggregate statistics and judge labels only are released.
- By default the harness persists only judge labels and aggregate statistics; raw completions are not written to disk. The opt-in encrypted-completions sink (Section 5.6) is available for reviewer-audit purposes under a documented Microsoft-managed key-management and access protocol; reviewers can request access via the program chairs.

### 8.3 Disclosure precedent
- We cite Anthropic's many-shot disclosure timeline as a contemporary precedent and the Microsoft RAI Standard for internal policy.
- Although we report no novel attack, our headline findings name production models from four vendors and use four public benchmarks/judges as instruments, so we follow a documented courtesy-disclosure procedure. A pre-print draft is shared with the four model providers (Meta, Microsoft, Google DeepMind, Alibaba) and the four benchmark/judge maintainers (HarmBench, XSTest, OR-Bench, Llama-Guard-3) 14-30 days before public release. Contact dates are logged in Appendix I alongside the pre-registration tag / commit URL.

### 8.4 Risk of highlighting weak baselines
- Ranking sub-4B models on safety risks unfairly labeling a model as "unsafe" when the differential is small or judge-dependent.
- Mitigations: report all judges, report confidence intervals on every claim, flag every ranking flip, avoid headline "safest model" phrasing in favor of "safest under judge X for attack family Y".

### 8.5 Author conflict of interest
- The author is at Microsoft and Phi-3 is a Microsoft model.
- Mitigations: public pre-registration of the protocol (committed and tagged in the public repository before any run); identical defensive prompt content across all four models (the placement asymmetry for Gemma-2, whose chat template does not accept a system role, is disclosed in Section 4.4 and ablated in Section 6.5); triangulated judges; explicit COI statement in this section.
- Pre-commitment: if Phi-3-mini is the safety-frontier model under any judge, we (i) report the result with the same emphasis and confidence treatment as for any other model, (ii) explicitly attempt to find a judge or condition under which it is not, and (iii) include the negative-result attempt in the paper. This mirrors the Microsoft Responsible AI Standard's requirement that comparative claims about Microsoft models be supported by reproducible measurement protocols and counter-claims be reported transparently.

### 8.6 Participant and data ethics
- No human subjects.
- All benchmark datasets are previously published with appropriate licenses, verified against each dataset's LICENSE file: HarmBench (code MIT; behaviour text under the HarmBench repository terms with provenance to component datasets), JBB-Behaviors (MIT), XSTest (CC-BY-4.0), OR-Bench (CC-BY-4.0), AdvBench (distributed within the GCG MIT-licensed repository, with provenance to prior harmful-behavior corpora noted).
- A license-compliance checklist is in Appendix E.

### 8.7 Environmental footprint
- Approximately 18 GPU-hours on a consumer GPU; approximately 0.3-0.5 kWh estimated.
- Exact kWh measurement is reported either from a Kill-A-Watt reading or from an `nvidia-smi`-derived integral, whichever is feasible at submission time.

**Key citations.** `zou2023gcg`, `mazeika2024harmbench`, `chao2024jailbreakbench`, `anil2024manyshot`.

---

## 9. Limitations (~700 words)

- **Static-attack scope.** Adaptive attacks (Andriushchenko et al.) reach 100% ASR on Phi-3-mini and similar; we do not measure adaptive robustness. This is the headline limitation and is restated in the abstract.
- **English-only.** Multilingual jailbreaks (Yong et al., Deng et al. MultiJail) are out of scope; sub-4B open models have heterogeneous multilingual coverage and conflating capability with safety would be misleading.
- **Single-turn only.** Multi-turn Crescendo-style attacks are out of scope.
- **Four-model scope.** Mistral-7B excluded (weaker safety documentation); 8B+ excluded (outside the sub-4B class); OLMo-1B and SmolLM noted as future work.
- **Single primary defensive prompt.** Variant sensitivity is reported in Section 6.5 but the headline numbers are for a single prompt.
- **Judge ceiling.** We deliberately exclude paid GPT-4 rubric judging (StrongREJECT-style) to preserve laptop reproducibility; this trades against absolute ASR accuracy.
- **Sample size.** 200 behaviors per cell yields ~7-point Wilson CI on rate estimates; category-level subdivisions are wider and we restrict claims accordingly.
- **FRR benchmark overlap.** XSTest and OR-Bench-hard partially overlap thematically; we report Spearman between them in Section 6.8.
- **Hugging Face revision drift.** Pinned but the underlying weights could change; the 50-prompt drift check in Appendix H bounds but does not eliminate this risk.
- **No closed-model comparison.** Scope is open laptop-deployable models; closed-model practitioners should consult the HarmBench and JBB leaderboards.

**Key citations.** `andriushchenko2025adaptive`, `yong2023lowresource`, `deng2024multilingual`, `souly2024strongreject`.

---

## 10. Conclusion (~400 words)

- Restate the practitioner-relevance framing: the on-device sub-4B class is the deployment frontier where audit infrastructure is thinnest.
- Restate the three closed gaps and the empirical headline findings: judge-dependent ranking, non-uniform defense cost, capability-driven safety artifacts for cipher attacks.
- Restate the non-contribution stance (no new attack, no new defense) as a feature for the Responsible AI audience.
- Pointer to the open harness and a one-line invocation that reproduces every table in the paper.
- Forward look: extending to multi-turn, multilingual, and the 1B-tier (SmolLM, OLMo-1B) as on-device LLMs continue to shrink.

**Key citations.** None.

---

## References

- Full BibTeX of all citations across foundational works, recent papers, and evaluation frameworks.
- ACM Reference Format for the FAccT submission; the arXiv preprint uses the same `.bib`.
- A build-script check verifies that every citation in the prose appears in the `.bib` and vice versa.

---

## Appendices

- **Appendix A.** Full reproducibility kit: Hugging Face commit hashes, `environment.yml`, single shell command, expected runtime per stage.
- **Appendix B.** Verbatim defensive system prompt (primary) and the two pre-registered variants (terse, constitutional-style).
- **Appendix C.** Full per-judge per-model per-condition tables with all confidence intervals and McNemar p-values.
- **Appendix D.** One-page practitioner checklist for auditing a sub-4B open model before deployment. The checklist opens with an explicit scope statement: it applies to single-turn English chat deployments of sub-4B open models behind a single defensive system prompt; it does *not* cover multi-turn manipulation, tool/agent use, RAG with attacker-controlled context, multilingual users, fine-tuning attacks, or indirect prompt injection (Greshake et al.). For those, the checklist points to the dedicated literature.
- **Appendix E.** License compliance and data card per Gebru et al.; model card per Mitchell et al. for the harness output.
- **Appendix F.** Judge calibration: agreement matrix between HarmBench classifier, Llama-Guard-3, and the keyword baseline on a held-out 100-sample subset.
- **Appendix G.** Prompt-template versioning notes per model (Phi-3 vs 3.5, Gemma-2 vs 2-it, etc.) and known chat-template pitfalls.
- **Appendix H.** Drift-check results across two Hugging Face revisions per model.
- **Appendix I.** Pre-registration tag / commit URL and a snapshot of the pre-registered protocol.

**Key citations.** `mazeika2024harmbench`, `inan2023llamaguard`, `rottger2024xstest`, `cui2024orbench`.

---

## Planned Figures

- **Figure 1.** Joint (ASR, FRR) frontier for the four sub-4B open models under empty and defensive system prompts, aggregated across three judges with shaded judge-variance bands; arrows show the within-model shift induced by the defensive prompt.
  - *Implementation.* Matplotlib scatter; x-axis ASR (HarmBench-200, mean across 3 judges), y-axis FRR (XSTest-250); two points per model connected by an arrow; shaded ellipse marks judge variance; color per model; legend includes the Andriushchenko adaptive-attack 100% line as a dashed ceiling for reference.

- **Figure 2.** Per-judge ASR with Wilson 95% CIs for each (model, system-prompt) cell, illustrating the judge-dependent ranking pattern.
  - *Implementation.* Grouped bar chart: 4 models x 2 prompts x 3 judges = 24 bars; error bars are Wilson CIs; ranking annotations above each judge group.

- **Figure 3.** Defense cost C(M) = Delta-FRR / Delta-ASR per model under the primary defensive system prompt; the >2x spread across models is the headline fleet-management finding.
  - *Implementation.* Horizontal bar chart; one bar per model; bootstrap B=1000 CI whiskers; annotate the ratio between max and min defense cost.

- **Figure 4.** Attack-family breakdown: ASR per HarmBench semantic category x model under the empty prompt; highlights capability-driven safety on cipher and encoded prompts and persistent vulnerability to DAN-family persona attacks.
  - *Implementation.* Heatmap with rows = 7 HarmBench semantic categories, columns = 4 models, cell value = ASR; the cipher row is annotated with a footnote on decoding-capability failure.

- **Figure 5.** Judge agreement matrix on a held-out 100-prompt calibration subset, showing pairwise agreement between HarmBench classifier, Llama-Guard-3-1B, and the keyword baseline.
  - *Implementation.* 3x3 heatmap of Cohen's kappa or percent-agreement; cells annotated with raw counts; placed in Appendix F but referenced from Section 4.5 and Section 6.3.

- **Figure 6.** System-prompt sensitivity: ASR and FRR dispersion across the primary and two pre-registered variant defensive prompts (terse, constitutional-style).
  - *Implementation.* Two panels (ASR, FRR); model on x-axis; three points per model, one per prompt variant; error bars are Wilson CIs.

---

## Novelty Statement

The paper's novelty is methodological and comparative rather than capability-extending, matching FAccT's preference for audit and measurement contributions. The literature-review coverage matrix in Section 2 lists each comparative-audit precedent against five axes (sub-4B class, joint ASR+FRR same models, system-prompt intervention measured, judge triangulation with ranking-stability test, laptop reproducibility) and shows that the five-axis intersection is empty in the published literature: HarmBench and JailbreakBench omit the sub-4B class; XSTest, OR-Bench, and PHTest measure FRR without paired ASR on the same models; Xie et al. and Zhang et al. evaluate system-prompt defenses on closed models without joint FRR audit; StrongREJECT and JailJudge document judge variance without applying it to a model-ranking decision; WildGuard and SORRY-Bench cover individual axes but not the intersection. Our reproducible laptop-runnable harness — pinned model revisions, three judges, joint metric, pre-registered defensive prompt and variants — is the first artifact that lets a practitioner re-run an audit of the on-device open-model class before deployment without GPU clusters or paid API budgets. The paper proposes no new attack and no new defense; the contribution is the evaluation protocol, the joint frontier, and the ranking-stability evidence.

---

## Risk Mitigations (cross-cutting)

- **Judge unreliability.** Triangulate across HarmBench classifier, Llama-Guard-3, and a keyword baseline; report Kendall's tau ranking stability; pre-register a flip-detection rule; cite Souly et al. and Lin et al. for the documented 30-point judge variance.
- **Model-version drift.** Pin Hugging Face commit hashes; document chat-template versions per model; run a 50-prompt drift check against an earlier checkpoint per model and bound the resulting ASR delta in Appendix H.
- **Narrow attack coverage.** Explicit threat-model-floor framing — we measure the static-attack distribution that approximates realistic paste-attackers; cite Andriushchenko et al. as the adaptive ceiling; never claim adaptive robustness.
- **Ethics of publicizing attack patterns.** Only public benchmarks; no inlined harmful content; aggregate-only reporting; encrypted-at-rest raw completions for reviewer audit; Microsoft RAI Standard followed; responsible-disclosure footnote.
- **Defending the choice of a weak baseline defense.** Position the single system prompt as the practitioner-deployable lower bound; cite Xie self-reminder (67%->19%) and Zhang goal-priority (66%->3.6%) as evidence that simple prompts move ASR substantially; frame stronger defenses (SmoothLLM, circuit breakers) as infrastructure-dependent and out of practitioner reach.
- **Single defensive prompt.** Pre-register the verbatim primary prompt plus two variants (terse, constitutional); report dispersion in Section 6.5.
- **Sample-size concerns.** Wilson 95% CIs, McNemar paired tests, B=1000 bootstrap for ASR-difference distributions; restrict category-level claims to those passing bootstrap stability.
- **"Small models trivially broken" critique.** Distinguish the static-attack distribution (our scope) from adaptive log-prob search (Andriushchenko); argue paste-attackers do not run log-prob search.
- **Open-only scope.** Explicit positioning as an on-device / air-gapped deployment audit; closed-model leaderboards referenced for completeness; framed as a scope contribution, not an omission.
- **OR-Bench / XSTest overlap.** XSTest designated primary, OR-Bench-hard secondary robustness check; Spearman correlation between the two reported.
- **Multi-turn omission.** Explicit single-turn scope; cite Crescendo-style literature as future work; argue single-turn results still inform the practitioner baseline.
- **Model-selection rationale.** Scope to sub-4B class with credible safety documentation; explicitly justify exclusion of Mistral-7B (safety doc) and Llama-3.1-8B (out of scope), with OLMo-1B and SmolLM as future work.
- **Author COI (Microsoft / Phi-3).** Public pre-registration of protocol (committed and tagged in the public repository before any run); identical defensive prompt across all four models; triangulated judges; explicit COI statement in Ethics section.

---

## Central Claims (pre-registered hypotheses)

These are stated as pre-registered hypotheses to be tested under the publicly pre-registered protocol, not as findings established at the time of writing. Section 6 reports the observed direction and magnitude of each.

1. **H1 (judge dependence).** Under a unified protocol, the safety ranking of the four sub-4B open models will not be stable across the three judges; at least one pairwise model ordering will flip between the HarmBench fine-tuned classifier and Llama-Guard-3-1B per the flip-detection rule in Section 5.7. If H1 holds, "safest small model" is a judge-dependent claim and single-judge ASR reports in the small-model space are insufficient for procurement decisions.
2. **H2 (defense cost spread).** A single ~60-word defensive system prompt will reduce ASR on all four models (effect sizes in the same order of magnitude as Xie et al.'s self-reminder result on ChatGPT), but the FRR penalty on XSTest and OR-Bench-Hard will be non-uniform across models, producing a model-specific (ASR-reduction, FRR-increase) trade-off curve whose spread across models is more than a factor of two by the defense-cost ratio C(M) and the area metric A(M) defined in Section 4.1.
3. **H3 (capability-vs-alignment safety).** Attack-family vulnerability will be differentiated across the four models in interpretable ways: encoding and cipher attacks (Yuan et al.) will under-perform on sub-4B models because the models fail to decode the ciphers at all, while DAN-family persona attacks will remain effective. If H3 holds, small-model "safety" partially derives from limited capability and will erode as small-model capability improves.
4. **Methodological contribution.** The reproducible, laptop-runnable harness with pinned model revisions, three-judge triangulation, and joint (ASR, FRR) reporting is a feasible and FAccT-aligned alternative to single-judge, single-axis evaluations for the practitioner audience. This is a contribution-by-construction (the harness itself), not a pre-registered hypothesis.
