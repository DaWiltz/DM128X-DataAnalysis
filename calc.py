from __future__ import annotations

import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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

INPUT_JSON: Path | None = None
OUT_DIR = Path("outputs")
PLOTS_DIR = OUT_DIR / "plots"

TEMPS = ["T=0", "T=0.35", "T=0.70"]

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

    _json_input = input("Enter path to input JSON file: ").strip()
    json_path = Path(_json_input)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON source not found: {json_path}")

    print(f"Loaded data from JSON: {json_path}")
    return pd.read_json(json_path)


_DEMO_COLS = {
    "participant_age": "age",
    "participant_gender": "gender",
    "participant_game_experience": "game_experience",
    "participant_native_language": "native_language",
}


def extract_demographics(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "temp" in df.columns and "temperature" not in df.columns:
        df = df.rename(columns={"temp": "temperature"})

    demo = df[df["q_id"].isin(_DEMO_COLS)].copy()
    if demo.empty:
        return pd.DataFrame(columns=["player", "temperature"] + list(_DEMO_COLS.values()))

    demo["temperature"] = demo["temperature"].apply(normalize_temperature)

    wide = demo.pivot_table(index="player", columns="q_id", values="answer", aggfunc="first")
    wide.columns.name = None
    wide = wide.rename(columns=_DEMO_COLS).reset_index()

    temp_map = demo.dropna(subset=["temperature"]).groupby("player")["temperature"].first()
    wide["temperature"] = wide["player"].map(temp_map)

    present = ["player", "temperature"] + [c for c in _DEMO_COLS.values() if c in wide.columns]
    wide["temperature"] = pd.Categorical(wide["temperature"], categories=TEMPS, ordered=True)
    return wide[present].sort_values(["temperature", "player"]).reset_index(drop=True)


_OPEN_COLS = {
    "game_experience_postplay_short_12": "immersion_moment",
    "game_experience_postplay_short_13": "immersion_break",
    "game_experience_postplay_short_14": "memorable_moment",
}


def extract_open_answers(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "temp" in df.columns and "temperature" not in df.columns:
        df = df.rename(columns={"temp": "temperature"})

    open_rows = df[df["q_id"].isin(_OPEN_COLS)].copy()
    if open_rows.empty:
        return pd.DataFrame(columns=["player", "temperature"] + list(_OPEN_COLS.values()))

    open_rows["temperature"] = open_rows["temperature"].apply(normalize_temperature)

    wide = open_rows.pivot_table(index="player", columns="q_id", values="answer", aggfunc="first")
    wide.columns.name = None
    wide = wide.rename(columns=_OPEN_COLS).reset_index()

    temp_map = open_rows.dropna(subset=["temperature"]).groupby("player")["temperature"].first()
    wide["temperature"] = wide["player"].map(temp_map)

    present = ["player", "temperature"] + [c for c in _OPEN_COLS.values() if c in wide.columns]
    wide["temperature"] = pd.Categorical(wide["temperature"], categories=TEMPS, ordered=True)
    return wide[present].sort_values(["temperature", "player"]).reset_index(drop=True)


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
            fallback: dict[str, Any] = {
                "factor": factor,
                "test": "insufficient_groups",
                "stat": np.nan,
                "df": "",
                "p": np.nan,
                "eta_sq": np.nan,
                "levene_stat": np.nan,
                "levene_p": np.nan,
                "shapiro_p_min": np.nan,
            }
            for temp in TEMPS:
                fallback[f"shapiro_p_{temp}"] = np.nan
            anova_rows.append(fallback)
            continue

        groups = list(valid.values())
        labels = list(valid.keys())
        n_total = int(sum(len(g) for g in groups))
        k_groups = len(groups)

        if all(len(g) >= 2 for g in groups):
            levene_stat, levene_p = levene(*groups, center="mean")
        else:
            levene_stat, levene_p = np.nan, np.nan

        shapiro_ps = []
        shapiro_p_by_temp: dict[str, float] = {}
        for label, g in zip(labels, groups):
            if len(g) >= 3:
                _, p_sh = shapiro(g)
                shapiro_ps.append(float(p_sh))
                shapiro_p_by_temp[label] = float(p_sh)
            else:
                shapiro_p_by_temp[label] = np.nan
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

        row: dict[str, Any] = {
            "factor": factor,
            "test": test_name,
            "stat": float(stat),
            "df": df_text,
            "p": float(p_val),
            "eta_sq": eta_sq,
            "levene_stat": float(levene_stat) if not np.isnan(levene_stat) else np.nan,
            "levene_p": float(levene_p) if not np.isnan(levene_p) else np.nan,
            "shapiro_p_min": float(shapiro_min) if not np.isnan(shapiro_min) else np.nan,
        }
        for temp in TEMPS:
            row[f"shapiro_p_{temp}"] = shapiro_p_by_temp.get(temp, np.nan)
        anova_rows.append(row)

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


def run_overall_test(scored: pd.DataFrame) -> pd.DataFrame:
    label_to_group = {
        temp: scored.loc[scored["temperature"] == temp, "Overall"].dropna().to_numpy(dtype=float)
        for temp in TEMPS
    }
    valid = {k: v for k, v in label_to_group.items() if len(v) > 0}

    if len(valid) < 2:
        row: dict[str, Any] = {
            "test": "insufficient_groups",
            "stat": np.nan,
            "df": "",
            "p": np.nan,
            "eta_sq": np.nan,
            "levene_stat": np.nan,
            "levene_p": np.nan,
            "shapiro_p_min": np.nan,
        }
        for temp in TEMPS:
            row[f"shapiro_p_{temp}"] = np.nan
        return pd.DataFrame([row])

    groups = list(valid.values())
    labels = list(valid.keys())
    n_total = int(sum(len(g) for g in groups))
    k_groups = len(groups)

    if all(len(g) >= 2 for g in groups):
        levene_stat, levene_p = levene(*groups, center="mean")
    else:
        levene_stat, levene_p = np.nan, np.nan

    shapiro_ps = []
    shapiro_p_by_temp: dict[str, float] = {}
    for label, g in zip(labels, groups):
        if len(g) >= 3:
            _, p_sh = shapiro(g)
            shapiro_ps.append(float(p_sh))
            shapiro_p_by_temp[label] = float(p_sh)
        else:
            shapiro_p_by_temp[label] = np.nan
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

    out: dict[str, Any] = {
        "test": test_name,
        "stat": float(stat),
        "df": df_text,
        "p": float(p_val),
        "eta_sq": eta_sq,
        "levene_stat": float(levene_stat) if not np.isnan(levene_stat) else np.nan,
        "levene_p": float(levene_p) if not np.isnan(levene_p) else np.nan,
        "shapiro_p_min": float(shapiro_min) if not np.isnan(shapiro_min) else np.nan,
    }
    for temp in TEMPS:
        out[f"shapiro_p_{temp}"] = shapiro_p_by_temp.get(temp, np.nan)

    return pd.DataFrame([out])


def build_conclusions(anova_df: pd.DataFrame, tukey_df: pd.DataFrame) -> str:
    lines = ["CONCLUSIONS", "=" * 60]

    sig = anova_df[anova_df["p"] < 0.05]
    nonsig = anova_df[anova_df["p"] >= 0.05]

    if sig.empty:
        lines.append(
            "No significant differences between temperature conditions were found for any factor (all p > 0.05)."
        )
        involvement_eta = (
            anova_df.loc[anova_df["factor"] == "Involvement", "eta_sq"].iloc[0]
            if "Involvement" in anova_df["factor"].values else float("nan")
        )
        if not np.isnan(involvement_eta):
            lines.append(
                f"Note: The current sample size may be insufficient to detect existing effects "
                f"(Involvement eta^2 = {involvement_eta:.2f} suggests a potentially large effect)."
            )
    else:
        for _, row in sig.iterrows():
            test_label = "one-way ANOVA" if row["test"].lower() == "anova" else "Kruskal-Wallis test"
            stat_label = "F" if row["test"].lower() == "anova" else "H"
            lines.append(
                f"\n{row['factor']}: Significant effect of temperature detected "
                f"({test_label}, {stat_label}({row['df']}) = {row['stat']:.3f}, "
                f"p = {row['p']:.3f}, eta^2 = {row['eta_sq']:.3f})."
            )
            if not tukey_df.empty:
                factor_pairs = tukey_df[tukey_df["factor"] == row["factor"]]
                sig_pairs = factor_pairs[factor_pairs["p"] < 0.05]
                if not sig_pairs.empty:
                    lines.append("  Post-hoc comparisons (Tukey HSD):")
                    for _, trow in sig_pairs.iterrows():
                        bonf = " [Bonferroni-significant]" if trow["sig_bonf"] else ""
                        lines.append(
                            f"    * {trow['comparison']}: delta_mean = {trow['delta_mean']:.3f}, "
                            f"p = {trow['p']:.3f}{bonf}"
                        )
                else:
                    lines.append("  No individual pairs reached significance in post-hoc tests.")

    if not nonsig.empty:
        lines.append("\nNon-significant factors:")
        for _, row in nonsig.iterrows():
            test_label = "one-way ANOVA" if row["test"].lower() == "anova" else "Kruskal-Wallis test"
            lines.append(
                f"  * {row['factor']}: {test_label}, p = {row['p']:.3f}, eta^2 = {row['eta_sq']:.3f}"
            )

    return "\n".join(lines)


def print_summary(descriptives: pd.DataFrame, anova_df: pd.DataFrame) -> None:
    print("\nDescriptives (factor x temperature):")
    print(descriptives.round(4).to_string(index=False))

    print("\nFactor tests:")
    display = anova_df.copy()
    display["sig"] = np.where(display["p"] < 0.05, "*", "")
    print(display.round(4).to_string(index=False))
