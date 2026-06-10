"""
Post-hoc analysis for the small-LLM jailbreak / over-refusal study.

Consumes the ``results.csv`` written by ``05_experiment.py`` and produces:

  1. ASR / FRR per (model, condition) with non-parametric bootstrap 95% CIs.
  2. Pareto frontier of ASR vs FRR per model (no_defense vs with_defense), with
     per-model dominance flags (does with_defense Pareto-dominate no_defense?).
  3. Ranking-stability analysis: Kendall's tau on model rankings by ASR and by
     FRR, comparing the two conditions (and the two judges if both are present).
  4. Pareto-front scatter plot across all (model, condition, judge) points,
     with each model's two conditions connected to make the tradeoff visible.
  5. Per-model bar chart of the safety / over-refusal tradeoff
     (delta_ASR = ASR_with_defense - ASR_no_defense, and the analogous
     delta_FRR), so a "good" defense shows a negative ASR bar and a small
     positive FRR bar.
  6. All auxiliary outputs (per-condition metrics, Pareto table, Kendall taus,
     plots) saved under ``<out_dir>/results/``.

Ethics note:
  - This script touches only aggregate metrics. It does not load, log, or
    re-emit any adversarial prompt content or model completion.
  - The bootstrap is computed from per-(model, condition) refusal *counts*
    reconstructed from the rates and sample sizes in ``results.csv``;
    no per-prompt outcomes are needed or persisted.
"""

# =============================================================================
# Imports and CLI
# =============================================================================

import argparse
import csv
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-csv",
        default="results.csv",
        help="Path to the results.csv produced by 05_experiment.py.",
    )
    p.add_argument(
        "--out-dir",
        default=".",
        help="Directory under which a 'results/' subdirectory is created for "
             "all auxiliary outputs (tables, plots, JSON summaries).",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for ASR/FRR difference intervals "
             "(pre-registered B=1000).",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Two-sided confidence level for bootstrap intervals (e.g. 0.95).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the bootstrap resampler.",
    )
    return p.parse_args()


# =============================================================================
# Loading and validation
# =============================================================================

# Per-prompt schema written by 05_experiment.py. This script aggregates
# internally; no separate aggregate file is required.
ROW_COLUMNS = [
    "model", "condition", "benchmark", "prompt_id", "judge",
    "verdict", "harmful", "refused",
]

# Aliases that prior versions of 05_experiment.py used. Loaded for backward
# compatibility but immediately re-mapped onto the current schema.
LEGACY_ALIASES = {
    "judge_used": "judge",
}


def load_results(path: str) -> pd.DataFrame:
    """Load the per-prompt results.csv written by 05_experiment.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"results.csv not found at {path}. Run 05_experiment.py first."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in LEGACY_ALIASES.items()
                            if k in df.columns})
    missing = [c for c in ROW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}. Re-run 05_experiment.py "
            "to refresh the schema."
        )
    for col in ("harmful", "refused"):
        df[col] = df[col].astype(int)
    return df


# Attack vs benign benchmark sets. Aggregation respects these.
_ATTACK_BENCHES = {"harmbench", "jbb", "harmbench_gcg"}
_BENIGN_BENCHES = {"xstest", "orbench"}


def aggregate_per_cell(rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-prompt rows to per (model, condition, judge) aggregates.

    Generation-error rows are excluded from both numerators and denominators
    and are reported as a separate column.
    """
    out: List[Dict] = []
    for (model, cond, judge), sub in rows.groupby(["model", "condition", "judge"]):
        att = sub[sub["benchmark"].isin(_ATTACK_BENCHES)]
        ben = sub[sub["benchmark"].isin(_BENIGN_BENCHES)]
        att_ok = att[att["verdict"] != "error"]
        ben_ok = ben[ben["verdict"] != "error"]
        asr = float(att_ok["harmful"].mean()) if len(att_ok) else float("nan")
        frr = float(ben_ok["refused"].mean()) if len(ben_ok) else float("nan")
        n_err = int((sub["verdict"] == "error").sum())
        n_tot = int(len(sub))
        out.append(dict(
            model=model, condition=cond, judge=judge,
            asr=asr, frr=frr,
            n_attack=int(len(att_ok)),
            n_benign=int(len(ben_ok)),
            error_rate=(n_err / n_tot) if n_tot else 0.0,
        ))
    return pd.DataFrame(out)


# =============================================================================
# Confidence intervals and paired tests
# =============================================================================

@dataclass
class RateCI:
    rate: float
    lo: float
    hi: float
    n: int
    successes: int


def _wilson_ci(successes: int, n: int, ci: float = 0.95) -> RateCI:
    """Wilson score interval for a Bernoulli rate (closed form)."""
    if n <= 0:
        return RateCI(rate=float("nan"), lo=float("nan"),
                      hi=float("nan"), n=0, successes=0)
    s = max(0, min(int(successes), int(n)))
    p = s / n
    alpha = 1.0 - ci
    # Normal critical value for two-sided CI.
    z = math.sqrt(2.0) * _erfinv(1.0 - alpha)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return RateCI(rate=p, lo=float(center - half),
                  hi=float(center + half), n=int(n), successes=int(s))


def _erfinv(x: float) -> float:
    """Approximate inverse error function (Winitzki's formula)."""
    a = 0.147
    s = math.copysign(1.0, x)
    ln = math.log(max(1e-12, 1.0 - x * x))
    term = 2.0 / (math.pi * a) + ln / 2.0
    return s * math.sqrt(math.sqrt(term * term - ln / a) - term)


def _paired_bootstrap_diff(
    pairs: List[Tuple[int, int]],
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Paired nonparametric bootstrap CI for a difference of rates.

    `pairs` is a list of (success_a, success_b) 0/1 tuples for the SAME
    prompts under two conditions. Resamples pairs with replacement and
    reports the (mean_a - mean_b) percentile interval. Returns
    (point_estimate, lo, hi).
    """
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(pairs, dtype=np.int8)
    a = arr[:, 0]
    b = arr[:, 1]
    diff = float(a.mean() - b.mean())
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return diff, lo, hi


def _mcnemar(pairs: List[Tuple[int, int]]) -> Tuple[float, int, int]:
    """Exact-binomial McNemar test on paired 0/1 outcomes.

    Returns (p_value, b, c) where b is the count of (1, 0) and c of (0, 1).
    The exact test uses the binomial distribution with n=b+c and p=0.5.
    """
    if not pairs:
        return float("nan"), 0, 0
    b = sum(1 for a, x in pairs if a == 1 and x == 0)
    c = sum(1 for a, x in pairs if a == 0 and x == 1)
    nbc = b + c
    if nbc == 0:
        return 1.0, 0, 0
    # Two-sided exact binomial p-value.
    k = min(b, c)
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(nbc, i) * (0.5 ** nbc)
    p = float(min(1.0, 2.0 * total))
    return p, b, c


def compute_metric_cis(
    df: pd.DataFrame,
    n_boot: int,
    ci: float,
    seed: int,
) -> pd.DataFrame:
    """Return per-cell aggregates with Wilson CIs on ASR and FRR.

    The aggregate is computed from the per-prompt rows (excluding generation
    errors). For each (model, condition, judge) we report ASR, FRR, sample
    sizes, and Wilson 95% intervals. n_boot is accepted for API stability
    but not used here (per-cell single-rate CIs are closed-form Wilson).
    """
    del seed  # not used; Wilson intervals are closed-form
    del n_boot
    cells = aggregate_per_cell(df)
    out_rows: List[Dict] = []
    for _, row in cells.iterrows():
        asr_succ = int(round(row["asr"] * row["n_attack"])) if row["n_attack"] else 0
        frr_succ = int(round(row["frr"] * row["n_benign"])) if row["n_benign"] else 0
        asr_ci = _wilson_ci(asr_succ, int(row["n_attack"]), ci)
        frr_ci = _wilson_ci(frr_succ, int(row["n_benign"]), ci)
        out_rows.append({
            "model": row["model"],
            "condition": row["condition"],
            "judge": row["judge"],
            "asr": asr_ci.rate,
            "asr_lo": asr_ci.lo,
            "asr_hi": asr_ci.hi,
            "frr": frr_ci.rate,
            "frr_lo": frr_ci.lo,
            "frr_hi": frr_ci.hi,
            "n_attack": asr_ci.n,
            "n_benign": frr_ci.n,
            "error_rate": float(row.get("error_rate", 0.0)),
        })
    return pd.DataFrame(out_rows)


def compute_paired_tests(
    df: pd.DataFrame,
    n_boot: int,
    ci: float,
    seed: int,
) -> pd.DataFrame:
    """Paired bootstrap + McNemar for within-model before/after defense.

    For each (model, judge) we pair per-prompt outcomes across the two
    conditions (no_defense, with_defense) for both ASR (HarmBench-side
    benchmarks) and FRR (XSTest/OR-Bench).
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for (model, judge), sub in df.groupby(["model", "judge"]):
        for metric, benches, col in (
            ("asr_diff", _ATTACK_BENCHES, "harmful"),
            ("frr_diff", _BENIGN_BENCHES, "refused"),
        ):
            cell = sub[sub["benchmark"].isin(benches)]
            cell = cell[cell["verdict"] != "error"]
            piv = cell.pivot_table(
                index=["benchmark", "prompt_id"], columns="condition",
                values=col, aggfunc="max",
            )
            if not {"no_defense", "with_defense"}.issubset(piv.columns):
                continue
            piv = piv.dropna(subset=["no_defense", "with_defense"])
            pairs = list(zip(piv["with_defense"].astype(int).tolist(),
                             piv["no_defense"].astype(int).tolist()))
            diff, lo, hi = _paired_bootstrap_diff(pairs, n_boot, ci, rng)
            p, b, c = _mcnemar(pairs)
            rows.append(dict(
                model=model, judge=judge, metric=metric,
                n_pairs=len(pairs),
                diff=diff, lo=lo, hi=hi,
                mcnemar_p=p, b_only_with=b, c_only_without=c,
            ))
    return pd.DataFrame(rows)


# =============================================================================
# Pareto frontier
# =============================================================================

@dataclass
class ParetoRow:
    model: str
    condition: str
    judge: str
    asr: float
    frr: float
    on_frontier: bool
    dominates_other_condition: bool


def _is_dominated(point: Tuple[float, float],
                  others: List[Tuple[float, float]]) -> bool:
    """Lower ASR and lower FRR are both better.

    A point is *dominated* if there exists another point that is no worse on
    either axis and strictly better on at least one.
    """
    x, y = point
    for ox, oy in others:
        if (ox <= x and oy <= y) and (ox < x or oy < y):
            return True
    return False


def compute_pareto(df_ci: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pareto frontier over (ASR, FRR) and per-model dominance."""
    points: List[Tuple[float, float]] = list(
        zip(df_ci["asr"].tolist(), df_ci["frr"].tolist())
    )

    rows: List[ParetoRow] = []
    # Per-model: does with_defense dominate no_defense (lower-or-equal on both,
    # strictly lower on at least one)? Computed per judge.
    dom_map: Dict[Tuple[str, str], bool] = {}
    for (model, judge), g in df_ci.groupby(["model", "judge"]):
        sub = g.set_index("condition")
        if {"no_defense", "with_defense"}.issubset(sub.index):
            nd = sub.loc["no_defense"]
            wd = sub.loc["with_defense"]
            dominates = (
                wd["asr"] <= nd["asr"] and wd["frr"] <= nd["frr"]
                and (wd["asr"] < nd["asr"] or wd["frr"] < nd["frr"])
            )
            dom_map[(model, judge)] = bool(dominates)

    for i, (_, row) in enumerate(df_ci.iterrows()):
        others = [p for j, p in enumerate(points) if j != i]
        dominated = _is_dominated((row["asr"], row["frr"]), others)
        rows.append(ParetoRow(
            model=row["model"],
            condition=row["condition"],
            judge=row["judge"],
            asr=row["asr"],
            frr=row["frr"],
            on_frontier=not dominated,
            dominates_other_condition=dom_map.get(
                (row["model"], row["judge"]), False
            ),
        ))
    return pd.DataFrame([asdict(r) for r in rows])


# =============================================================================
# Kendall's tau (ranking stability)
# =============================================================================

def _kendall_tau(x: List[float], y: List[float]) -> Tuple[float, int, int]:
    """Kendall's tau-b on paired numeric ranks.

    Returns (tau, concordant, discordant). NaN if fewer than 2 pairs or no
    informative variation. We implement this directly to avoid pulling in
    SciPy for a single statistic.
    """
    n = len(x)
    if n < 2:
        return float("nan"), 0, 0
    concordant = 0
    discordant = 0
    tx = 0  # ties in x only
    ty = 0  # ties in y only
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue  # tied on both axes -- contributes nowhere
            if dx == 0:
                tx += 1
                continue
            if dy == 0:
                ty += 1
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom_x = concordant + discordant + tx
    denom_y = concordant + discordant + ty
    if denom_x == 0 or denom_y == 0:
        return float("nan"), concordant, discordant
    tau = (concordant - discordant) / math.sqrt(denom_x * denom_y)
    return tau, concordant, discordant


def compute_ranking_stability(df_ci: pd.DataFrame) -> Dict[str, Dict]:
    """Compute Kendall's tau between condition rankings (and judge rankings).

    Two stability questions are answered:
      (a) Across models, does the ASR ranking under no_defense agree with the
          ASR ranking under with_defense? (And the same for FRR.)
          A high tau means model ordering is robust to the defense; a low or
          negative tau means the defense reshuffles which model "wins".
      (b) If both judges are present, does the ranking under the keyword
          judge agree with the ranking under Llama-Guard?
    """
    out: Dict[str, Dict] = {}

    judges = sorted(df_ci["judge"].unique().tolist())
    conditions = ["no_defense", "with_defense"]

    # (a) condition-vs-condition, per judge, for each metric
    cond_stab: Dict[str, Dict[str, Dict]] = {}
    for judge in judges:
        sub = df_ci[df_ci["judge"] == judge]
        cond_stab[judge] = {}
        for metric in ("asr", "frr"):
            pivot = sub.pivot_table(
                index="model", columns="condition", values=metric
            )
            if set(conditions).issubset(pivot.columns) and len(pivot) >= 2:
                pivot = pivot.dropna(subset=conditions)
                x = pivot["no_defense"].tolist()
                y = pivot["with_defense"].tolist()
                tau, conc, disc = _kendall_tau(x, y)
                cond_stab[judge][metric] = {
                    "tau": tau,
                    "concordant": conc,
                    "discordant": disc,
                    "n_models": len(pivot),
                    "models": pivot.index.tolist(),
                }
            else:
                cond_stab[judge][metric] = {
                    "tau": float("nan"), "concordant": 0, "discordant": 0,
                    "n_models": 0, "models": [],
                }
    out["condition_stability"] = cond_stab

    # (b) judge-vs-judge, per condition, for each metric, for EVERY pair of
    # judges. Three judges therefore yield three pairwise taus rather than one;
    # this is the cross-judge ranking-stability evidence the paper reports for
    # H1, and it must cover all pairs (keyword/Llama-Guard/HarmBench), not just
    # the first two encountered.
    judge_stab: Dict[str, Dict[str, Dict]] = {}
    for j1, j2 in itertools.combinations(judges, 2):
        pair_label = f"{j1}__vs__{j2}"
        judge_stab[pair_label] = {}
        for cond in conditions:
            judge_stab[pair_label][cond] = {}
            for metric in ("asr", "frr"):
                sub = df_ci[df_ci["condition"] == cond]
                pivot = sub.pivot_table(
                    index="model", columns="judge", values=metric
                )
                if {j1, j2}.issubset(pivot.columns) and len(pivot) >= 2:
                    pivot = pivot.dropna(subset=[j1, j2])
                    tau, conc, disc = _kendall_tau(
                        pivot[j1].tolist(), pivot[j2].tolist()
                    )
                    judge_stab[pair_label][cond][metric] = {
                        "tau": tau,
                        "concordant": conc,
                        "discordant": disc,
                        "n_models": len(pivot),
                        "judges": [j1, j2],
                        "models": pivot.index.tolist(),
                    }
                else:
                    judge_stab[pair_label][cond][metric] = {
                        "tau": float("nan"), "concordant": 0, "discordant": 0,
                        "n_models": 0, "judges": [j1, j2], "models": [],
                    }
    out["judge_stability"] = judge_stab

    return out


# =============================================================================
# Deltas (safety vs over-refusal tradeoff per model)
# =============================================================================

def compute_deltas(df_ci: pd.DataFrame) -> pd.DataFrame:
    """Per (model, judge), the with_defense - no_defense delta on ASR and FRR.

    A 'good' defense is delta_asr <= 0 (safety improves or holds) with
    delta_frr small and non-negative (over-refusal cost is contained).
    """
    rows: List[Dict] = []
    for (model, judge), g in df_ci.groupby(["model", "judge"]):
        sub = g.set_index("condition")
        if not {"no_defense", "with_defense"}.issubset(sub.index):
            continue
        nd = sub.loc["no_defense"]
        wd = sub.loc["with_defense"]
        rows.append({
            "model": model,
            "judge": judge,
            "asr_no_defense": float(nd["asr"]),
            "asr_with_defense": float(wd["asr"]),
            "delta_asr": float(wd["asr"] - nd["asr"]),
            "frr_no_defense": float(nd["frr"]),
            "frr_with_defense": float(wd["frr"]),
            "delta_frr": float(wd["frr"] - nd["frr"]),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

# Distinct, colorblind-friendly per-model palette. Falls back to matplotlib's
# default cycle if more models than colors are present.
_MODEL_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#8c564b",  # brown
]


def _color_for(model: str, model_order: List[str]) -> str:
    idx = model_order.index(model)
    return _MODEL_COLORS[idx % len(_MODEL_COLORS)]


def plot_pareto(df_ci: pd.DataFrame, pareto_df: pd.DataFrame, path: str) -> None:
    """ASR (x) vs FRR (y) scatter, one point per (model, condition, judge).

    Per-model lines connect no_defense -> with_defense within the same judge.
    Points on the global Pareto frontier are circled.
    """
    model_order = sorted(df_ci["model"].unique().tolist())
    judges = sorted(df_ci["judge"].unique().tolist())
    # Distinct markers per judge so a paper-grade reader can separate them.
    judge_markers = {j: m for j, m in zip(judges, ["o", "s", "D", "^"])}

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # Per-model connecting lines (no_defense -> with_defense) per judge.
    for model in model_order:
        color = _color_for(model, model_order)
        for judge in judges:
            sub = df_ci[(df_ci["model"] == model) & (df_ci["judge"] == judge)]
            sub = sub.set_index("condition")
            if {"no_defense", "with_defense"}.issubset(sub.index):
                xs = [sub.loc["no_defense", "asr"], sub.loc["with_defense", "asr"]]
                ys = [sub.loc["no_defense", "frr"], sub.loc["with_defense", "frr"]]
                ax.plot(xs, ys, "-", color=color, alpha=0.4, linewidth=1.2)

    # Scatter points, with frontier markers highlighted.
    frontier_lookup = {
        (r["model"], r["condition"], r["judge"]): r["on_frontier"]
        for _, r in pareto_df.iterrows()
    }
    for _, row in df_ci.iterrows():
        color = _color_for(row["model"], model_order)
        marker = judge_markers.get(row["judge"], "o")
        on_front = frontier_lookup.get(
            (row["model"], row["condition"], row["judge"]), False
        )
        face = color if row["condition"] == "with_defense" else "white"
        edge = color
        ax.scatter(
            row["asr"], row["frr"],
            s=110 if on_front else 70,
            marker=marker,
            facecolors=face,
            edgecolors=edge,
            linewidths=2.0 if on_front else 1.2,
            zorder=3,
        )

    ax.set_xlabel("ASR (lower is safer)")
    ax.set_ylabel("FRR (lower is more helpful on benign)")
    ax.set_title("Pareto frontier: safety vs over-refusal\n"
                 "filled = with_defense, open = no_defense; "
                 "large outline = on global Pareto frontier")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Legend: one entry per model (color), one per judge (marker).
    from matplotlib.lines import Line2D
    handles = []
    for model in model_order:
        handles.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=_color_for(model, model_order),
            markeredgecolor=_color_for(model, model_order),
            markersize=9, label=model,
        ))
    for judge, marker in judge_markers.items():
        handles.append(Line2D(
            [0], [0], marker=marker, color="black",
            markerfacecolor="none", markeredgecolor="black",
            linestyle="", markersize=9, label=f"judge={judge}",
        ))
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tradeoff_bars(deltas: pd.DataFrame, path: str) -> None:
    """Grouped bars per model: delta_ASR (lower=better) and delta_FRR (cost).

    If both judges are present, a separate subplot is produced per judge so the
    comparison is apples-to-apples within a refusal definition.
    """
    if deltas.empty:
        # Nothing to plot; emit a placeholder so downstream tooling sees a file.
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No (no_defense, with_defense) pairs available",
                ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    judges = sorted(deltas["judge"].unique().tolist())
    fig, axes = plt.subplots(
        1, len(judges),
        figsize=(max(7.0, 3.5 * len(judges) + 2.0), 4.5),
        sharey=True,
        squeeze=False,
    )
    for j_idx, judge in enumerate(judges):
        ax = axes[0][j_idx]
        sub = deltas[deltas["judge"] == judge].copy()
        sub = sub.sort_values("delta_asr")  # most-improved on safety first
        models = sub["model"].tolist()
        x = np.arange(len(models))
        width = 0.38
        ax.bar(x - width / 2, sub["delta_asr"].values, width,
               label="ΔASR (negative = safer)", color="#1f77b4")
        ax.bar(x + width / 2, sub["delta_frr"].values, width,
               label="ΔFRR (positive = more over-refusal)", color="#d62728")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(f"judge = {judge}")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        if j_idx == 0:
            ax.set_ylabel("with_defense - no_defense")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Per-model safety / over-refusal tradeoff from the defense")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# Writers
# =============================================================================

def write_df_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, float_format="%.4f",
              quoting=csv.QUOTE_MINIMAL)


def write_json(obj: Dict, path: str) -> None:
    def _default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON-serializable: {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"[load] reading {args.results_csv}", flush=True)
    df = load_results(args.results_csv)
    print(f"[load] rows={len(df)} models={df['model'].nunique()} "
          f"conditions={df['condition'].nunique()} "
          f"judges={df['judge'].nunique()}", flush=True)

    # ---- (1) Wilson 95% CIs on per-cell ASR / FRR
    print("[ci] computing Wilson 95% intervals on per-cell rates ...",
          flush=True)
    df_ci = compute_metric_cis(df, args.bootstrap, args.ci, args.seed)
    cis_path = os.path.join(results_dir, "metrics_with_ci.csv")
    write_df_csv(df_ci, cis_path)
    print(f"[ci] wrote {cis_path}", flush=True)

    # ---- (1b) paired nonparametric bootstrap + McNemar
    print(f"[paired] computing paired bootstrap (B={args.bootstrap}) "
          f"and McNemar p-values for within-model before/after defense ...",
          flush=True)
    paired = compute_paired_tests(df, args.bootstrap, args.ci, args.seed)
    paired_path = os.path.join(results_dir, "paired_tests.csv")
    write_df_csv(paired, paired_path)
    print(f"[paired] wrote {paired_path}", flush=True)

    # ---- (2) Pareto frontier
    pareto_df = compute_pareto(df_ci)
    pareto_path = os.path.join(results_dir, "pareto.csv")
    write_df_csv(pareto_df, pareto_path)
    print(f"[pareto] frontier size = "
          f"{int(pareto_df['on_frontier'].sum())} / {len(pareto_df)} points "
          f"-> {pareto_path}", flush=True)

    # ---- (3) Kendall's tau ranking stability
    stab = compute_ranking_stability(df_ci)
    stab_path = os.path.join(results_dir, "ranking_stability.json")
    write_json(stab, stab_path)
    for judge, m in stab.get("condition_stability", {}).items():
        for metric, info in m.items():
            tau = info.get("tau", float("nan"))
            print(f"[tau] judge={judge} metric={metric.upper()} "
                  f"no_defense-vs-with_defense tau={tau:.3f} "
                  f"(n={info.get('n_models', 0)})", flush=True)
    for pair_label, conds in stab.get("judge_stability", {}).items():
        for cond, m in conds.items():
            for metric, info in m.items():
                tau = info.get("tau", float("nan"))
                print(f"[tau] judges={pair_label} condition={cond} "
                      f"metric={metric.upper()} tau={tau:.3f} "
                      f"(n={info.get('n_models', 0)})", flush=True)

    # ---- (4) Pareto scatter plot
    pareto_png = os.path.join(results_dir, "pareto_scatter.png")
    plot_pareto(df_ci, pareto_df, pareto_png)
    print(f"[plot] wrote {pareto_png}", flush=True)

    # ---- (5) per-model tradeoff bar chart
    deltas = compute_deltas(df_ci)
    deltas_path = os.path.join(results_dir, "deltas.csv")
    write_df_csv(deltas, deltas_path)
    bar_png = os.path.join(results_dir, "tradeoff_bars.png")
    plot_tradeoff_bars(deltas, bar_png)
    print(f"[plot] wrote {bar_png} (deltas table: {deltas_path})", flush=True)

    # ---- (6) one-shot summary for the paper's results section
    summary = {
        "n_rows": int(len(df)),
        "models": sorted(df["model"].unique().tolist()),
        "conditions": sorted(df["condition"].unique().tolist()),
        "judges": sorted(df["judge"].unique().tolist()),
        "bootstrap": {
            "n_resamples": int(args.bootstrap),
            "ci_level": float(args.ci),
            "seed": int(args.seed),
        },
        "pareto_frontier_size": int(pareto_df["on_frontier"].sum()),
        "defense_dominates_per_model": [
            {"model": r["model"], "judge": r["judge"],
             "dominates_other_condition": bool(r["dominates_other_condition"])}
            for _, r in pareto_df.drop_duplicates(
                subset=["model", "judge"]
            ).iterrows()
        ],
        "ranking_stability": stab,
        "outputs": {
            "metrics_with_ci": os.path.relpath(cis_path, args.out_dir),
            "paired_tests": os.path.relpath(paired_path, args.out_dir),
            "pareto_table": os.path.relpath(pareto_path, args.out_dir),
            "ranking_stability_json": os.path.relpath(stab_path, args.out_dir),
            "pareto_scatter_png": os.path.relpath(pareto_png, args.out_dir),
            "tradeoff_bars_png": os.path.relpath(bar_png, args.out_dir),
            "deltas_csv": os.path.relpath(deltas_path, args.out_dir),
        },
    }
    summary_path = os.path.join(results_dir, "summary.json")
    write_json(summary, summary_path)
    print(f"[done] wrote summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        raise SystemExit(2)
