"""Smoke test: 05_experiment.py output schema is consumable by 06_analysis.py.

Writes a synthetic results.csv with 4 prompts under the schema written by
05_experiment.py, then loads it with 06_analysis.load_results and runs the
aggregation / Wilson CI / paired-bootstrap path to confirm no schema errors.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import pandas as pd


_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_smoke():
    analysis = _import(
        "analysis_under_test", os.path.join(_HERE, "06_analysis.py")
    )

    rows = []
    for model in ("modelA", "modelB"):
        for cond in ("no_defense", "with_defense"):
            for jud in ("keyword", "llamaguard"):
                # 2 HarmBench, 2 XSTest prompts per (model, condition, judge).
                rows += [
                    dict(model=model, condition=cond, benchmark="harmbench",
                         prompt_id="h1", judge=jud, verdict="complied",
                         harmful=1 if cond == "no_defense" else 0, refused=0),
                    dict(model=model, condition=cond, benchmark="harmbench",
                         prompt_id="h2", judge=jud, verdict="refused",
                         harmful=0, refused=0),
                    dict(model=model, condition=cond, benchmark="xstest",
                         prompt_id="x1", judge=jud, verdict="complied",
                         harmful=0, refused=0),
                    dict(model=model, condition=cond, benchmark="xstest",
                         prompt_id="x2", judge=jud, verdict="refused",
                         harmful=0, refused=1 if cond == "with_defense" else 0),
                ]

    with tempfile.TemporaryDirectory() as tmp:
        results_path = os.path.join(tmp, "results.csv")
        pd.DataFrame(rows).to_csv(results_path, index=False)

        df = analysis.load_results(results_path)
        assert len(df) == len(rows)

        cells = analysis.aggregate_per_cell(df)
        assert set(cells.columns) >= {
            "model", "condition", "judge", "asr", "frr",
            "n_attack", "n_benign", "error_rate",
        }

        cis = analysis.compute_metric_cis(df, n_boot=10, ci=0.95, seed=0)
        assert {"asr_lo", "asr_hi", "frr_lo", "frr_hi"}.issubset(cis.columns)

        paired = analysis.compute_paired_tests(df, n_boot=50, ci=0.95, seed=0)
        # 2 models * 2 judges * 2 metrics = 8 rows when pairs exist.
        assert len(paired) >= 1


if __name__ == "__main__":
    test_pipeline_smoke()
    print("smoke test passed")
