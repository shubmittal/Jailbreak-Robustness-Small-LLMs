"""pin_revisions.py

Freeze the pre-registration by resolving the current ``main``-branch commit
SHA of every Hugging Face repository the audit harness touches: the four
models under test, the judge repositories, and the four benchmark datasets.
The author runs this once, pastes the resulting SHAs into ``DEFAULT_MODELS`` /
the ``*_DATASETS`` specs in ``05_experiment.py`` and into
``08_preregistration.md``, and thereby pins the weights and data to immutable
revisions before tagging the pre-registration.

This is a *metadata-only* utility. It uses ``huggingface_hub.HfApi``'s
``model_info`` / ``dataset_info`` calls with ``revision="main"`` and reads only
the returned ``.sha``. It NEVER downloads model weights or dataset files, never
instantiates a model, and never loads any prompt content. Consistent with the
audit-not-capability posture of this project, no benchmark prompt, jailbreak
string, or model completion is read or emitted here -- only repository ids and
commit hashes.

Repository ids are imported verbatim from ``05_experiment.py`` (the single
source of truth) rather than re-hardcoded here, so the frozen set can never
drift from the set the harness actually loads.

The four models, the judge repositories -- Llama-Guard-3-1B
[@inan2023llamaguard] plus the two HarmBench fine-tuned classifier variants
(the large Llama-2-13B and the small Mistral-7B-val) [@mazeika2024harmbench];
the keyword baseline judge needs no repository -- and four datasets (HarmBench
[@mazeika2024harmbench], XSTest [@rottger2024xstest], JailbreakBench, OR-Bench
[@cui2024orbench]) are the artifacts named in the pre-registration. (Note: the
three judge *methods* in the harness are the keyword baseline, Llama-Guard, and
the HarmBench classifier; the HarmBench classifier resolves to two pinned
repos, so this utility pins three judge repositories for two model-backed
judges.) Pinning their revisions supports the reproducibility commitments
expected of model cards [@mitchell2019modelcards] and datasheets
[@gebru2021datasheets].

Usage:
    python pin_revisions.py

Output:
    - Prints one ``repo  ->  sha`` line per repository (or a clear GATED
      message on a 401/403), grouped by role.
    - Writes the resolved mapping to ``revisions.json`` in the working
      directory. Resolved commit SHAs live under a ``"resolved"`` section
      (a clean repo->SHA mapping safe to paste); any gated/errored repos are
      kept separately under ``"unresolved"`` so a non-SHA message can never be
      mistaken for a revision and pasted into a revision slot.

Gated repositories (several of the model and judge repos require accepting a
license) are reported per-repo and never abort the rest of the run; resolve
them with ``huggingface-cli login`` and by accepting the license at the printed
URL, then re-run.
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple

# Single source of truth: pull the exact repo ids from the harness so the
# pinned set cannot drift from the set actually loaded at run time. We import
# only constant tuples/strings; importing this module does not run the harness
# (it is guarded by its own __main__) and triggers no downloads.
from importlib import import_module


def _load_experiment_module():
    """Import 05_experiment.py and return the module object.

    ``05_experiment`` is not a valid Python identifier, so a plain
    ``import_module("05_experiment")`` normally fails with ModuleNotFoundError
    and we load the file by path instead. We are careful NOT to swallow a
    ModuleNotFoundError that is really about a *missing third-party dependency*
    of the harness (e.g. the harness's own ``import tqdm`` failing): re-running
    that same module via the file-path loader would just raise the same error
    from deeper in the import machinery and obscure the real diagnosis. So we
    only fall back when the failing module is literally ``05_experiment``.
    """
    try:
        return import_module("05_experiment")
    except ModuleNotFoundError as exc:
        # Only treat a *missing 05_experiment module* as the trigger for the
        # file-path fallback. A missing dependency referenced by the harness
        # (different module name) is a real error and is re-raised.
        if exc.name not in (None, "05_experiment"):
            raise
    except ImportError:
        # Any other import-time failure of "05_experiment" -> try the path
        # loader, which will surface the underlying error if it persists.
        pass

    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_exp_05", os.path.join(here, "05_experiment.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not locate 05_experiment.py next to pin_revisions.py. "
            "Run this utility from the project root."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_exp = _load_experiment_module()


OUTPUT_PATH = "revisions.json"


def _collect_repos() -> Tuple[List[str], List[str], List[str]]:
    """Return (model_repos, judge_repos, dataset_repos), read from the harness.

    - Models come from DEFAULT_MODELS (first element of each tuple).
    - Judge repos are LLAMAGUARD_MODEL_ID plus the two HARMBENCH_CLS_* repos
      (the large Llama-2-13B and small Mistral-7B-val HarmBench classifiers).
      The keyword baseline judge has no repository and is therefore not pinned.
    - Datasets are gathered by iterating EVERY (repo, config, split, revision)
      spec in each of the four *_DATASETS tuples. The harness's
      _try_load_dataset accepts a list of fallback specs, so we pin all of
      them (de-duplicated) rather than only the first, in case a fallback
      mirror is ever added.
    """
    model_repos = [repo for repo, _rev in _exp.DEFAULT_MODELS]

    judge_repos = [
        _exp.LLAMAGUARD_MODEL_ID,
        _exp.HARMBENCH_CLS_LARGE[0],
        _exp.HARMBENCH_CLS_SMALL[0],
    ]

    dataset_repos: List[str] = []
    for dataset_specs in (
        _exp.HARMBENCH_DATASETS,
        _exp.XSTEST_DATASETS,
        _exp.JBB_DATASETS,
        _exp.ORBENCH_DATASETS,
    ):
        for spec in dataset_specs:
            repo = spec[0]
            if repo not in dataset_repos:
                dataset_repos.append(repo)

    return model_repos, judge_repos, dataset_repos


def _is_gated_error(exc: Exception) -> bool:
    """Heuristically detect a 401/403/gated-access failure from huggingface_hub."""
    # Prefer a structured HTTP status if huggingface_hub surfaces one.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    s = str(exc).lower()
    return (
        "401" in s
        or "403" in s
        or "gated" in s
        or "awaiting a review" in s
        or "access to model" in s
        or "access to dataset" in s
        or "you are trying to access a gated" in s
        or "authentication" in s
        or "unauthorized" in s
        or "forbidden" in s
    )


def _gated_message(repo: str, kind: str) -> str:
    """Build the actionable GATED message for a model/judge or dataset repo."""
    if kind == "dataset":
        url = f"https://huggingface.co/datasets/{repo}"
    else:
        url = f"https://huggingface.co/{repo}"
    return (
        "GATED: run `huggingface-cli login` and accept the license at "
        f"{url}"
    )


def resolve_sha(api, repo: str, kind: str) -> str:
    """Resolve the ``main`` commit SHA for one repo using metadata only.

    ``kind`` is one of {"model", "judge", "dataset"}; models and judges live in
    the model namespace, datasets in the dataset namespace. On a gated/auth
    failure a GATED message string is returned (the call is caught here so one
    gated repo never aborts the rest). Other unexpected errors are surfaced as
    an ``ERROR: ...`` string, again without aborting the batch.
    """
    try:
        if kind == "dataset":
            info = api.dataset_info(repo, revision="main")
        else:
            info = api.model_info(repo, revision="main")
        sha = getattr(info, "sha", None)
        if not sha:
            return "ERROR: huggingface_hub returned no commit SHA for main."
        return sha
    except Exception as exc:  # noqa: BLE001 - one bad repo must not abort the rest
        if _is_gated_error(exc):
            return _gated_message(repo, kind)
        return f"ERROR: {type(exc).__name__}: {exc}"


def main() -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "huggingface_hub is required: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 2

    api = HfApi()

    model_repos, judge_repos, dataset_repos = _collect_repos()

    # Preserve grouping for human-readable output; (repo, kind) drives resolve.
    groups: List[Tuple[str, List[Tuple[str, str]]]] = [
        ("MODELS (paste into DEFAULT_MODELS)",
         [(r, "model") for r in model_repos]),
        ("JUDGES (Llama-Guard-3-1B + the two HarmBench classifier repos)",
         [(r, "judge") for r in judge_repos]),
        ("DATASETS (paste into the *_DATASETS specs)",
         [(r, "dataset") for r in dataset_repos]),
    ]

    # Keep clean repo->SHA pins separate from gated/errored entries so the
    # JSON the author pastes from can never contain a non-SHA "revision".
    resolved: Dict[str, str] = {}
    unresolved: Dict[str, str] = {}
    gated_count = 0
    error_count = 0

    print("Resolving main-branch commit SHAs (metadata only; no downloads).\n")
    for title, repos in groups:
        print(f"== {title} ==")
        for repo, kind in repos:
            result = resolve_sha(api, repo, kind)
            print(f"{repo}  ->  {result}")
            if result.startswith("GATED:"):
                unresolved[repo] = result
                gated_count += 1
            elif result.startswith("ERROR:"):
                unresolved[repo] = result
                error_count += 1
            else:
                resolved[repo] = result
        print()

    output = {"resolved": resolved, "unresolved": unresolved}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"Wrote {OUTPUT_PATH}: {len(resolved)} resolved SHA(s) under "
        f"\"resolved\", {len(unresolved)} unresolved repo(s) under "
        "\"unresolved\"."
    )

    if gated_count or error_count:
        print(
            f"Note: {gated_count} repo(s) were gated and {error_count} hit "
            "other errors; they are listed under \"unresolved\" (with a "
            "message, NOT a SHA) and must not be pasted into a revision slot. "
            "Authenticate / accept licenses and re-run to complete the pin."
        )

    print(
        "\nNext steps: paste each SHA from the \"resolved\" section into the "
        "matching entry of DEFAULT_MODELS and the *_DATASETS specs in "
        "05_experiment.py (replacing the placeholder \"main\" / None "
        "revisions), and record the same SHAs in 08_preregistration.md before "
        "tagging the pre-registration. Pinning to immutable commit SHAs -- "
        "not the moving \"main\" branch -- is what makes the frozen artifact "
        "reproducible."
    )

    # Exit non-zero only on genuine (non-gated) errors so CI/pre-flight checks
    # can distinguish "needs a login" from "something is broken".
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
