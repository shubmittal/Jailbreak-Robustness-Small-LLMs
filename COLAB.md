# Running the audit on Google Colab

> **Order matters.** Run this **only after** the pre-registration is public —
> pushed to GitHub and tagged (`prereg-v1`). The run writes
> `run_manifest.json` with a UTC timestamp, and that timestamp **must come after**
> the registration timestamp. If the run predates registration, the pre-registered
> claims (H1/H2) must be downgraded from "findings" to "hypotheses tested"
> (paper §5.7). Registering first is the whole point.

Use a **GPU runtime**: Runtime → Change runtime type → **T4 GPU**.
The full grid is ~12–18 GPU-hours. Free Colab disconnects around 12h, so use
**Colab Pro** or split the run across the steps below (each step is restartable).

---

### 0. (Recommended) Throwaway smoke test — confirm it runs on Colab
This is **not** the registered run (tiny, ungated model, discard the output):
```python
!python 05_experiment.py --models Qwen/Qwen2.5-3B-Instruct --judges keyword \
    --n 2 --max_new_tokens 16 --no_plot --output_dir ./results/smoke
```

### 1. Clone the *registered* snapshot
```python
REPO_URL = "https://github.com/<your-username>/Jailbreak-Robustness-Small-LLMs.git"
!git clone {REPO_URL} repo
%cd repo
!git checkout prereg-v1   # the exact pre-registered commit — runs the frozen code
```

### 2. Install dependencies
```python
!pip install -q -r requirements.txt
```

### 3. Hugging Face login (for the gated models)
First accept the licenses on each model page (Llama-3.2-3B, Gemma-2-2B,
Llama-Guard-3-1B), then:
```python
from huggingface_hub import notebook_login
notebook_login()   # paste a token that has access to the three gated models
```

### 4. Primary grid — keyword + Llama-Guard judges (~6 GPU-h)
```python
!python 05_experiment.py --backend transformers --n 200 \
    --judges keyword,llamaguard --defense primary \
    --output_dir ./results/primary_grid
```

### 5. HarmBench-classifier judge — separate pass (~3 GPU-h)
```python
!python 05_experiment.py --backend transformers --n 200 \
    --judges harmbench --harmbench-cls-size large --defense primary \
    --output_dir ./results/harmbench_judge
```
If the 13B classifier runs out of memory on a T4, use `--harmbench-cls-size small`.

### 6. Analysis (tables + figures)
```python
!python 06_analysis.py --results-csv ./results/primary_grid/results.csv \
    --out-dir ./results/primary_grid --bootstrap 1000 --ci 0.95
```

### 7. Save results before the VM resets (Colab is ephemeral!)
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/jailbreak_results
!cp -r ./results/* /content/drive/MyDrive/jailbreak_results/
```

---

**Sanity check after the run:** open `results/*/run_manifest.json` and confirm its
`timestamp` is **later** than your pre-registration tag (`prereg-v1`) push time. That single
check is what backs the "registered before running" claim.
