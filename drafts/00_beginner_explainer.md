# Beginner's Explainer: Small Models, Same Rules

## One-sentence summary

This paper measures how safely four small open language models — the kind that now run on laptops and phones — refuse harmful requests, how often they wrongly refuse harmless requests, and how much a single safety paragraph added to the system prompt actually helps.

## What is a language model, briefly

A large language model (LLM) is a neural network trained to predict the next word given everything that came before. Once trained, it can hold a conversation, write code, summarise documents, and follow instructions. The models we study — Llama-3.2-3B-Instruct (Meta), Phi-3-mini (Microsoft), Qwen2.5-3B-Instruct (Alibaba), and Gemma-2-2B-it (Google DeepMind) — each have between two and four billion parameters. By recent standards that is small. Their weights are openly available, they fit in the memory of a single consumer laptop GPU, and they increasingly ship inside on-device assistants and offline enterprise tools.

## What is a jailbreak

A "jailbreak" is a prompt designed to make a language model produce content it was trained to refuse. The intended refusals cover a familiar list: instructions for violence, weapons, abuse of children, fraud, malware, and so on. Jailbreaks come in many shapes. Some are role-play prompts ("pretend you are an AI without restrictions"). Some are obfuscations (write the request in a cipher, or in base64, or in a low-resource language). Some are adversarial strings of nonsense characters computed by an optimisation procedure to flip the model's refusal behaviour. The research community has built standard datasets of such prompts so that different models can be compared on the same examples. We use prompts from one of these datasets, **HarmBench**, and we never reproduce any of the actual prompt text in the paper.

## What is alignment

Alignment is the engineering process that teaches a base model to be helpful, honest, and harmless. The dominant recipe is supervised fine-tuning followed by reinforcement learning from human feedback (RLHF), with newer variants like Direct Preference Optimization. The output of this process is the "instruction-tuned" model the user interacts with. Alignment is what gives a model its default tendency to refuse a request for, say, instructions for a weapon. It is also what makes the model occasionally over-cautious.

## What is over-refusal

Over-refusal — sometimes called "exaggerated safety" or "false refusal" — happens when a model declines a request that is actually benign. The classic example is "how do I kill a Python process?" A naive safety filter trained on the word "kill" will refuse. A thoughtful model will answer the programming question. Over-refusal is a real cost. It makes assistants annoying. It makes them less useful in technical, medical, and legal domains where the surface vocabulary often overlaps with harmful topics. The research community has built standard datasets for measuring over-refusal too. We use **XSTest**, a set of 250 carefully written safe prompts, and **OR-Bench-Hard**, a larger set of about 1,300 prompts that look unsafe but are not.

## The safety-versus-over-refusal trade-off

A model can trivially achieve zero attack success by refusing every request. It can also achieve zero over-refusal by complying with every request. Neither extreme is acceptable. The interesting question is the shape of the trade-off between the two. We report both numbers together rather than separately:

- **Attack Success Rate (ASR).** The fraction of harmful prompts on which the model produces substantive harmful content. Lower is safer.
- **False Refusal Rate (FRR).** The fraction of benign prompts on which the model refuses anyway. Lower is more helpful.

A useful defense reduces ASR by more than it raises FRR. A bad defense reduces both numbers, but raises FRR by more than it reduces ASR — making the model safer in name only, at the cost of being less useful in practice.

## Why small models matter

Most published safety evaluations target models with seven billion or more parameters, or frontier closed models accessed through paid APIs. But the models that actually ship on laptops, phones, and air-gapped enterprise stacks are smaller. They run through tools like Ollama and llama.cpp. They are the substrate of on-device assistants. And in those settings, deployers rarely have access to the elaborate defensive infrastructure (auxiliary classifiers, decoding-time interventions, representation-level edits) that the published defense literature assumes. The one defense every deployer can apply is a natural-language **system prompt** — a short paragraph prepended to every conversation that tells the model how to behave. The audit gap we close is the joint measurement of safety and over-refusal on these small models under exactly that practitioner-grade defense.

## What we measure

For each of the four models, under two conditions (no defense, with a defensive system prompt), we run:

- Harmful prompts from HarmBench (and, optionally, JailbreakBench and HarmBench prompts with a publicly available adversarial suffix appended).
- Benign-but-suspicious prompts from XSTest and OR-Bench-Hard.

Each model output is then scored by three independent judges:

- A **keyword judge** that flags obvious refusal phrases. Deliberately weak; included as a methodological floor.
- **Llama-Guard-3-1B**, a small open safety classifier from Meta.
- The **HarmBench classifier**, a fine-tuned Llama-2-13B model (or its smaller Mistral-7B sibling on memory-constrained hardware).

We never let one judge fall back to another. We treat them as three independent measurements and report when they disagree.

## How to read the results

Three habits make the results easier to interpret.

**First, look at ASR and FRR together.** A model with low ASR and low FRR is doing well on both axes. A model with low ASR but high FRR is safe-by-refusing-everything. A model with low FRR but high ASR is helpful-by-saying-yes-to-anything. Procurement decisions need both numbers.

**Second, look across judges.** Different safety judges disagree by 10-30 percentage points on the same model output, depending on the rubric. A model that looks safest under the HarmBench classifier may not look safest under Llama-Guard. We report a statistic called Kendall's tau to summarise how stable the ranking is across judges, and we flag any case where the safest-model designation changes.

**Third, look at the defensive prompt as a trade-off, not a free lunch.** Adding a safety paragraph typically reduces attack success. It also typically raises over-refusal. The interesting quantity is the ratio: how much FRR did we pay for each unit of ASR we bought? We expect this ratio to differ across models — meaning a single organisational safety policy will hit different models differently.

## What this paper does not do

The paper proposes no new attack and no new defense. It does not measure adaptive adversaries (researchers who can tailor a fresh optimisation per model). It does not cover multi-turn conversations, non-English prompts, or tool-using agents. It is an **audit**: a careful, reproducible measurement of an existing situation, on a single consumer laptop. We describe what we found and what a practitioner should do with it.

## The bottom line

Small open language models are now the default substrate for on-device AI. A practitioner shipping one of them needs to know both how it refuses harmful requests and how it handles harmless ones, under the only defense most deployments can actually apply. That joint measurement is the contribution. The rest of the paper documents the protocol, the statistics, and the practitioner checklist that follows from it.
