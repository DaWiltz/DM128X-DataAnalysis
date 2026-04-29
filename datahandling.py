from __future__ import annotations

import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import levene
from scipy.stats import shapiro
from scipy.stats import tukey_hsd

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_JSON = Path("neo4j_query_table_data_2026-4-29(3).json")
OUT_DIR = Path("outputs")
PLOTS_DIR = OUT_DIR / "plots"

TEMPS = ["T=0", "T=0.35", "T=0.70"]

# Keep this mapping from your input configuration.
RAW_ITEM_TO_Q = {
    "game_experience_postplay_short_1": 5,
    "game_experience_postplay_short_2": 6,
    "game_experience_postplay_short_3": 7,
    "game_experience_postplay_short_4": 8,
    "game_experience_postplay_short_5": 1,
    "game_experience_postplay_short_6": 2,
    "game_experience_postplay_short_7": 3,
    "game_experience_postplay_short_8": 4,
    "game_experience_postplay_short_9": 9,
    "game_experience_postplay_short_10": 10,
    "game_experience_postplay_short_11": 11,
}

# Optional fallback map for JSON exports that do not include temperature.
# Example: {"Dr. Seacat": "T=0.35"}
PLAYER_TEMPERATURE_MAP: dict[str, str] = {}

REVERSED_ITEMS = [5, 10]

FACTORS = {
    "Involvement": [1, 2, 3, 4],
    "RWD": [5, 6, 7, 8],
    "Challenge": [9, 10, 11],
}

TEMP_COLORS = {"T=0": "#4C72B0", "T=0.35": "#DD8452", "T=0.70": "#55A868"}
TEMP_LABELS = {"T=0": "T = 0", "T=0.35": "T = 0.35", "T=0.70": "T = 0.70"}

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Override this query if needed via NEO4J_QUERY.
DEFAULT_NEO4J_QUERY = """
MATCH (u)-[:HAS_PLAYER]->(p)-[:ANSWERED]->(fa)-[:FOR_QUESTION]->(fq)-[:IN_FORM]->(f)
WHERE datetime(fa.created_at) >= datetime() - duration('P1D')
OPTIONAL MATCH (p)-[:HAS_TEMPERATURE_CONFIG]->(tc)
RETURN
  u.name AS user,
  p.name AS player,
  f.name_en AS form,
  coalesce(fq.question_id, fq.id, fq.name) AS q_id,
  fq.question_en AS question,
  fa.raw_answer AS answer,
  fa.value_type AS answer_type,
  fa.created_at AS submitted_at,
  coalesce(tc.temperature, tc.temperature_label, tc.name, tc.value) AS temperature
""".strip()


def normalize_temperature(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None

    # Numeric comparison handles float precision and pandas nullable types
    try:
        fval = float(value)
        if abs(fval) < 0.001:
            return "T=0"
        if abs(fval - 0.35) < 0.01:
            return "T=0.35"
        if abs(fval - 0.70) < 0.01:
            return "T=0.70"
    except (ValueError, TypeError):
        pass

    text = str(value).strip().lower().replace(" ", "")
    mapping = {
        "t=0": "T=0",
        "0": "T=0",
        "0.0": "T=0",
        "t=0.35": "T=0.35",
        "0.35": "T=0.35",
        "t=0.70": "T=0.70",
        "t=0.7": "T=0.70",
        "0.70": "T=0.70",
        "0.7": "T=0.70",
    }
    return mapping.get(text)


def parse_created_at(value: Any) -> pd.Timestamp:
    if isinstance(value, dict):
        try:
            return pd.Timestamp(
                year=int(value.get("year", 1970)),
                month=int(value.get("month", 1)),
                day=int(value.get("day", 1)),
                hour=int(value.get("hour", 0)),
                minute=int(value.get("minute", 0)),
                second=int(value.get("second", 0)),
            )
        except Exception:
            return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def question_to_qnum(question_id: Any) -> int | None:
    if question_id is None:
        return None

    qid = str(question_id).strip()

    if qid in RAW_ITEM_TO_Q:
        return RAW_ITEM_TO_Q[qid]

    m_short = re.search(r"game_experience_postplay_short_(\d+)$", qid)
    if m_short:
        q_num = int(m_short.group(1))
        if 1 <= q_num <= 11:
            return q_num
        return None

    m_qid = re.search(r"q0?([1-9]|1[01])(?:_|$)", qid, flags=re.IGNORECASE)
    if m_qid:
        return int(m_qid.group(1))

    return None


def fetch_neo4j_data() -> pd.DataFrame:
    if GraphDatabase is None:
        raise RuntimeError("neo4j package is not installed. Install with: pip install neo4j")

    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        raise RuntimeError("Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD to run Neo4j ingestion")

    query = os.getenv("NEO4J_QUERY", DEFAULT_NEO4J_QUERY)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            rows = session.run(query).data()
        return pd.DataFrame(rows)
    finally:
        driver.close()


def load_source_data() -> pd.DataFrame:
    try:
        if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
            df = fetch_neo4j_data()
            if not df.empty:
                print("Loaded data from Neo4j")
                return df
            print("Neo4j query returned no rows, falling back to JSON")
    except Exception as exc:
        print(f"Neo4j ingestion failed ({exc}), falling back to JSON")

    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"JSON source not found: {INPUT_JSON}")

    print(f"Loaded data from JSON: {INPUT_JSON}")
    return pd.read_json(INPUT_JSON)


def standardize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_candidates = {
        "u.name": "user",
        "p.name": "player",
        "f.name_en": "form",
        "fq.question_en": "question",
        "fa.raw_answer": "answer",
        "fa.value_type": "answer_type",
        "fa.created_at": "submitted_at",
        "question_id": "q_id",
        "fq.question_id": "q_id",
        "temp": "temperature",
        "llm_temperature": "temperature",
        "player_temperature": "temperature",
        "temperature_group": "temperature",
    }
    available = {k: v for k, v in rename_candidates.items() if k in df.columns}
    df = df.rename(columns=available)

    required = ["player", "q_id", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from source data: {missing}")

    if "submitted_at" not in df.columns:
        df["submitted_at"] = pd.NaT
    if "temperature" not in df.columns:
        df["temperature"] = pd.NA

    return df


def build_ieq_long(df: pd.DataFrame) -> pd.DataFrame:
    work = standardize_source_columns(df).copy()

    work["q_num"] = work["q_id"].apply(question_to_qnum)
    work = work[work["q_num"].between(1, 11, inclusive="both")].copy()

    work["raw_score"] = pd.to_numeric(work["answer"], errors="coerce")
    work = work.dropna(subset=["raw_score"])

    work["submitted_ts"] = work["submitted_at"].apply(parse_created_at)

    work["temperature"] = work["temperature"].apply(normalize_temperature)

    if PLAYER_TEMPERATURE_MAP:
        fallback_temp = work["player"].map(PLAYER_TEMPERATURE_MAP).apply(normalize_temperature)
        work["temperature"] = work["temperature"].fillna(fallback_temp)

    latest = (
        work.sort_values("submitted_ts")
        .groupby(["player", "q_num"], as_index=False)
        .tail(1)
    )

    temp_map = (
        latest.dropna(subset=["temperature"])
        .sort_values("submitted_ts")
        .groupby("player", as_index=False)
        .tail(1)
        .set_index("player")["temperature"]
        .to_dict()
    )

    latest["temperature"] = latest["player"].map(temp_map)

    missing_temp_players = sorted(latest[latest["temperature"].isna()]["player"].unique().tolist())
    if missing_temp_players:
        print(f"Skipping players with unrecognized temperature: {', '.join(missing_temp_players)}")

    latest = latest[latest["temperature"].isin(TEMPS)].copy()
    return latest


def pivot_ieq(latest: pd.DataFrame) -> pd.DataFrame:
    wide = latest.pivot_table(
        index=["player", "temperature"],
        columns="q_num",
        values="raw_score",
        aggfunc="first",
    )

    wide.columns = [f"Q{int(c)}" for c in wide.columns]

    for q in range(1, 12):
        col = f"Q{q}"
        if col not in wide.columns:
            wide[col] = np.nan

    wide = wide.reset_index()
    wide = wide[["player", "temperature"] + [f"Q{i}" for i in range(1, 12)]]
    return wide


def score_ieq(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    for item in REVERSED_ITEMS:
        col = f"Q{item}"
        scored[col] = 6 - pd.to_numeric(scored[col], errors="coerce")

    for factor, items in FACTORS.items():
        scored[factor] = scored[[f"Q{i}" for i in items]].mean(axis=1)

    scored["Overall"] = scored[[f"Q{i}" for i in range(1, 12)]].mean(axis=1)
    return scored


def compute_descriptives(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FACTORS:
        grp = scored.groupby("temperature")[factor]
        stats_df = grp.agg(["mean", "std", "count", "min", "max"]).reset_index()
        for _, row in stats_df.iterrows():
            rows.append(
                {
                    "factor": factor,
                    "temperature": row["temperature"],
                    "mean": row["mean"],
                    "sd": row["std"],
                    "n": int(row["count"]),
                    "min": row["min"],
                    "max": row["max"],
                }
            )
    out = pd.DataFrame(rows)
    out["temperature"] = pd.Categorical(out["temperature"], categories=TEMPS, ordered=True)
    return out.sort_values(["factor", "temperature"])


def eta_squared_anova(groups: list[np.ndarray]) -> float:
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    total = ss_between + ss_within
    if total == 0:
        return 0.0
    return float(ss_between / total)


def eta_squared_kruskal(h_stat: float, n_total: int, k_groups: int) -> float:
    denom = n_total - k_groups
    if denom <= 0:
        return 0.0
    val = (h_stat - k_groups + 1) / denom
    return float(max(0.0, min(1.0, val)))


def run_factor_tests(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anova_rows = []
    tukey_rows = []

    for factor in FACTORS:
        label_to_group = {
            temp: scored.loc[scored["temperature"] == temp, factor].dropna().to_numpy(dtype=float)
            for temp in TEMPS
        }
        valid = {k: v for k, v in label_to_group.items() if len(v) > 0}

        if len(valid) < 2:
            anova_rows.append(
                {
                    "factor": factor,
                    "test": "insufficient_groups",
                    "stat": np.nan,
                    "df": "",
                    "p": np.nan,
                    "eta_sq": np.nan,
                    "levene_p": np.nan,
                    "shapiro_p_min": np.nan,
                }
            )
            continue

        groups = list(valid.values())
        labels = list(valid.keys())
        n_total = int(sum(len(g) for g in groups))
        k_groups = len(groups)

        if all(len(g) >= 2 for g in groups):
            _, levene_p = levene(*groups, center="mean")
        else:
            levene_p = np.nan

        shapiro_ps = []
        for g in groups:
            if len(g) >= 3:
                _, p_sh = shapiro(g)
                shapiro_ps.append(float(p_sh))
        shapiro_min = min(shapiro_ps) if shapiro_ps else np.nan

        assumptions_hold = bool(
            (not np.isnan(levene_p))
            and (levene_p >= 0.05)
            and shapiro_ps
            and all(p >= 0.05 for p in shapiro_ps)
        )

        if assumptions_hold:
            stat, p_val = f_oneway(*groups)
            eta_sq = eta_squared_anova(groups)
            test_name = "anova"
            df_text = f"{k_groups - 1}, {n_total - k_groups}"
        else:
            stat, p_val = kruskal(*groups)
            eta_sq = eta_squared_kruskal(float(stat), n_total, k_groups)
            test_name = "kruskal"
            df_text = f"{k_groups - 1}"

        anova_rows.append(
            {
                "factor": factor,
                "test": test_name,
                "stat": float(stat),
                "df": df_text,
                "p": float(p_val),
                "eta_sq": eta_sq,
                "levene_p": float(levene_p) if not np.isnan(levene_p) else np.nan,
                "shapiro_p_min": float(shapiro_min) if not np.isnan(shapiro_min) else np.nan,
            }
        )

        if p_val < 0.05:
            tukey = tukey_hsd(*groups)
            idx_by_label = {label: i for i, label in enumerate(labels)}

            for a, b in combinations(TEMPS, 2):
                if a in idx_by_label and b in idx_by_label:
                    i = idx_by_label[a]
                    j = idx_by_label[b]
                    p_pair = float(tukey.pvalue[i, j])
                else:
                    p_pair = np.nan

                mean_a = float(np.mean(label_to_group[a])) if len(label_to_group[a]) else np.nan
                mean_b = float(np.mean(label_to_group[b])) if len(label_to_group[b]) else np.nan

                tukey_rows.append(
                    {
                        "factor": factor,
                        "comparison": f"{a} vs {b}",
                        "delta_mean": mean_a - mean_b,
                        "p": p_pair,
                        "sig_bonf": bool((not np.isnan(p_pair)) and (p_pair < 0.017)),
                    }
                )

    anova_df = pd.DataFrame(anova_rows)
    tukey_df = pd.DataFrame(tukey_rows)
    return anova_df, tukey_df


def plot_factor_figures(scored: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)

    for factor in FACTORS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("white")

        plot_data = scored.copy()
        plot_data["temperature"] = pd.Categorical(
            plot_data["temperature"], categories=TEMPS, ordered=True
        )
        plot_data = plot_data[plot_data["temperature"].isin(TEMPS)]

        stats_df = (
            plot_data.groupby("temperature", observed=True)[factor]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
        )
        stats_df["se"] = stats_df["sd"] / np.sqrt(stats_df["n"].clip(lower=1))
        stats_df = stats_df.sort_values("temperature").reset_index(drop=True)

        # ── Left panel: mean ± SE bar chart ──────────────────────────
        x = np.arange(len(stats_df))
        colors = [TEMP_COLORS.get(str(t), "#888888") for t in stats_df["temperature"]]

        bars = axes[0].bar(
            x, stats_df["mean"],
            yerr=stats_df["se"],
            capsize=6, width=0.5,
            color=colors, edgecolor="white", linewidth=0.8,
            error_kw={"elinewidth": 1.8, "ecolor": "#444444", "capthick": 1.8},
            zorder=2,
        )
        for bar, (_, row) in zip(bars, stats_df.iterrows()):
            top = bar.get_height() + row["se"] + 0.08
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, max(top, 1.12),
                f"{row['mean']:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333",
            )

        tick_labels = [
            f"{TEMP_LABELS.get(str(t), str(t))}\n(n = {int(row['n'])})"
            for t, (_, row) in zip(stats_df["temperature"], stats_df.iterrows())
        ]
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tick_labels, fontsize=10)
        axes[0].set_ylim(1, 5.7)
        axes[0].set_ylabel("Score (1–5)", fontsize=11)
        axes[0].set_title("Mean ± SE", fontsize=12, fontweight="bold", pad=10)
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[0].yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        axes[0].set_axisbelow(True)

        # ── Right panel: individual scores + mean crossbar ───────────
        sns.stripplot(
            data=plot_data, x="temperature", y=factor,
            hue="temperature", hue_order=TEMPS, palette=TEMP_COLORS,
            legend=False, order=TEMPS,
            alpha=0.85, size=8, jitter=0.12,
            ax=axes[1], zorder=3,
        )
        for i, temp in enumerate(TEMPS):
            grp = plot_data.loc[plot_data["temperature"] == temp, factor].dropna()
            if len(grp):
                axes[1].plot(
                    [i - 0.22, i + 0.22], [grp.mean(), grp.mean()],
                    color=TEMP_COLORS.get(temp, "#888888"),
                    linewidth=2.8, solid_capstyle="round", zorder=4,
                )

        axes[1].set_ylim(0.5, 5.5)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")
        axes[1].set_xticks(range(len(TEMPS)))
        axes[1].set_xticklabels([TEMP_LABELS.get(t, t) for t in TEMPS], fontsize=11)
        axes[1].set_title("Individual Scores  (— mean)", fontsize=12, fontweight="bold", pad=10)
        axes[1].spines[["top", "right"]].set_visible(False)
        axes[1].yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        axes[1].set_axisbelow(True)

        fig.suptitle(f"IEQ-SF — {factor}", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{factor.lower()}_by_temperature.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_tukey_summary(tukey_df: pd.DataFrame) -> None:
    if tukey_df.empty:
        return

    factor_order = [f for f in FACTORS if f in tukey_df["factor"].values]
    if not factor_order:
        return

    pivot = (
        tukey_df.pivot(index="factor", columns="comparison", values="p")
        .reindex(index=factor_order)
    )

    fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns) * 2.8), max(3, len(factor_order) * 1.4 + 1.5)))
    fig.patch.set_facecolor("white")

    mask = pivot.isna()
    sns.heatmap(
        pivot, ax=ax,
        cmap="RdYlGn_r", vmin=0, vmax=1,
        annot=True, fmt=".3f", annot_kws={"size": 10, "weight": "bold"},
        linewidths=1.5, linecolor="white",
        mask=mask,
        cbar_kws={"label": "p-value", "shrink": 0.75},
    )

    # Mark Bonferroni-significant cells
    for i, factor in enumerate(factor_order):
        for j, comp in enumerate(pivot.columns):
            p = pivot.loc[factor, comp]
            if not pd.isna(p) and p < 0.017:
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="#2d6a2d", linewidth=2.5)
                )

    ax.set_title(
        "Tukey HSD Post-hoc p-values\n(bold border = Bonferroni significant, α = 0.017)",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Comparison", fontsize=11, labelpad=8)
    ax.set_ylabel("Factor", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", rotation=15, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tukey_pvalues_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(descriptives: pd.DataFrame, anova_df: pd.DataFrame) -> None:
    print("\nDescriptives (factor x temperature):")
    print(descriptives.round(4).to_string(index=False))

    print("\nFactor tests:")
    display = anova_df.copy()
    display["sig"] = np.where(display["p"] < 0.05, "*", "")
    print(display.round(4).to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_source_data()
    ieq_long = build_ieq_long(raw_df)
    wide = pivot_ieq(ieq_long)
    scored = score_ieq(wide)

    descriptives = compute_descriptives(scored)
    anova_df, tukey_df = run_factor_tests(scored)

    anova_df.to_csv(OUT_DIR / "anova_results.csv", index=False)
    tukey_df.to_csv(OUT_DIR / "tukey_results.csv", index=False)
    descriptives.to_csv(OUT_DIR / "descriptives.csv", index=False)

    scored.to_csv(OUT_DIR / "participant_scores_q1_q11.csv", index=False)

    plot_factor_figures(scored)
    plot_tukey_summary(tukey_df)

    print_summary(descriptives, anova_df)
    print(f"\nSaved CSV outputs to: {OUT_DIR.resolve()}")
    print(f"Saved plots to: {PLOTS_DIR.resolve()}")
    print(f"Participants processed: {scored['player'].nunique()}")


if __name__ == "__main__":
    main()
