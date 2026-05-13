from __future__ import annotations

import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from calc import (
    FACTORS,
    PLOTS_DIR,
    REVERSED_ITEMS,
    TEMP_COLORS,
    TEMP_LABELS,
    TEMPS,
)

FACTOR_DESCRIPTIONS = {
    "Involvement": "Engagement & absorption during the game",
    "RWD":         "Real World Dissociation",
    "Challenge":   "Mental challenge & stimulation",
    "Overall":     "Mean across all 11 IEQ-SF items",
}

_SIG_COLORS = {
    "***": "#1a7a1a",
    "**":  "#2e9e2e",
    "*":   "#5cb85c",
    "ns":  "#aaaaaa",
}


def _setup_theme() -> None:
    sns.set_theme(style="ticks", font_scale=1.05)
    plt.rcParams.update({
        "axes.facecolor":  "#f8f9fa",
        "axes.edgecolor":  "#cccccc",
        "axes.linewidth":  0.8,
        "grid.color":      "white",
        "grid.linewidth":  1.5,
        "xtick.color":     "#555555",
        "ytick.color":     "#555555",
        "text.color":      "#222222",
        "font.size":       10,
    })


def _caption(fig: plt.Figure, text: str, y: float = 0.005) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=8, color="#999999", style="italic")


def _darken(hex_color: str, factor: float = 0.65) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(int(r*factor), int(g*factor), int(b*factor))


# ── Summary figure ─────────────────────────────────────────────────────────────

def plot_summary_figure(scored: pd.DataFrame) -> None:
    _setup_theme()

    factors_all = list(FACTORS.keys()) + ["Overall"]
    rng = np.random.default_rng(42)

    plot_data = scored.copy()
    plot_data["temperature"] = pd.Categorical(
        plot_data["temperature"], categories=TEMPS, ordered=True
    )
    plot_data = plot_data[plot_data["temperature"].isin(TEMPS)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    fig.patch.set_facecolor("white")

    for ax, factor in zip(axes.flat, factors_all):
        stats_df = (
            plot_data.groupby("temperature", observed=True)[factor]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
            .sort_values("temperature")
            .reset_index(drop=True)
        )
        stats_df["se"] = stats_df["sd"] / np.sqrt(stats_df["n"].clip(lower=1))

        x      = np.arange(len(stats_df))
        colors = [TEMP_COLORS.get(str(t), "#888888") for t in stats_df["temperature"]]
        darks  = [_darken(c, 0.68) for c in colors]

        ax.set_facecolor("#f8f9fa")
        ax.yaxis.grid(True, color="white", linewidth=1.5, zorder=0)
        ax.set_axisbelow(True)

        # Bars
        ax.bar(x, stats_df["mean"], width=0.5,
               color=colors, edgecolor=darks, linewidth=0.8,
               alpha=0.85, zorder=3)

        # Error bars
        for xi, dark, (_, row) in zip(x, darks, stats_df.iterrows()):
            ax.errorbar(xi, row["mean"], yerr=row["se"],
                        fmt="none", capsize=4, elinewidth=1.5, capthick=1.5,
                        ecolor=dark, zorder=4)

        # Individual dots
        for xi, temp in enumerate(TEMPS):
            grp = plot_data.loc[plot_data["temperature"] == temp, factor].dropna().values
            if len(grp):
                jitter = rng.uniform(-0.13, 0.13, len(grp))
                ax.scatter(xi + jitter, grp,
                           color=TEMP_COLORS[temp], s=22, alpha=0.75,
                           edgecolors="white", linewidth=0.4, zorder=5)

        # Mean labels
        for xi, dark, (_, row) in zip(x, darks, stats_df.iterrows()):
            ax.text(xi, row["mean"] + row["se"] + 0.08,
                    f"{row['mean']:.2f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=dark, zorder=6)

        ax.axhline(3, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_ylim(1, 5.6)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.tick_params(axis="y", labelsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{TEMP_LABELS[str(t)]}\n(n={int(r['n'])})"
             for t, (_, r) in zip(stats_df["temperature"], stats_df.iterrows())],
            fontsize=8.5,
        )

        ax.set_title(factor, fontsize=11, fontweight="bold", pad=6, color="#111111")
        ax.text(0.5, 1.01, FACTOR_DESCRIPTIONS.get(factor, ""),
                transform=ax.transAxes, ha="center",
                fontsize=7.5, color="#888888")

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")

    # Shared y-axis label
    fig.text(0.01, 0.5, "Score (1–5)", va="center", ha="left",
             rotation="vertical", fontsize=10, color="#555555")

    # Legend
    handles = [
        mpatches.Patch(facecolor=TEMP_COLORS[t], edgecolor=_darken(TEMP_COLORS[t], 0.7),
                       label=TEMP_LABELS[t], alpha=0.85)
        for t in TEMPS
    ]
    fig.legend(handles=handles, title="LLM Temperature",
               title_fontsize=9, fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=True, framealpha=0.9,
               edgecolor="#dddddd")

    fig.suptitle("IEQ-SF Scores by LLM Temperature Condition",
                 fontsize=13, fontweight="bold", y=1.01, color="#111111")

    fig.tight_layout(rect=[0.04, 0.09, 1.0, 0.98])
    fig.savefig(PLOTS_DIR / "factor_summary.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Tukey heatmap ──────────────────────────────────────────────────────────────

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

    fig, ax = plt.subplots(
        figsize=(max(6, len(pivot.columns) * 2.8), max(2.5, len(factor_order) * 1.3 + 1.5))
    )
    fig.patch.set_facecolor("white")

    sns.heatmap(
        pivot, ax=ax,
        cmap="RdYlGn_r", vmin=0, vmax=1,
        annot=True, fmt=".3f",
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=2.0, linecolor="white",
        mask=pivot.isna(),
        cbar_kws={"label": "p-value", "shrink": 0.8},
    )

    for i, factor in enumerate(factor_order):
        for j, comp in enumerate(pivot.columns):
            p = pivot.loc[factor, comp]
            if not pd.isna(p) and p < 0.017:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor="#1a6b1a", linewidth=2.5
                ))

    ax.set_title("Tukey HSD Post-hoc p-values",
                 fontsize=12, fontweight="bold", pad=12, color="#111111")
    ax.set_xlabel("Comparison", fontsize=10, labelpad=6)
    ax.set_ylabel("Factor", fontsize=10, labelpad=6)
    ax.tick_params(axis="x", rotation=15, labelsize=9.5)
    ax.tick_params(axis="y", rotation=0, labelsize=9.5)

    _caption(fig, "Green = lower p-value (stronger evidence of difference).  "
             "Bold border = Bonferroni-significant (α = 0.017).")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tukey_pvalues_summary.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── ANOVA table ────────────────────────────────────────────────────────────────

def _anova_table_figure(display: pd.DataFrame, title: str, filename: str) -> None:
    display = display.copy()
    display["test"] = (
        display["test"].str.upper()
        .replace("KRUSKAL", "Kruskal-Wallis")
        .replace("ANOVA", "One-way ANOVA")
    )
    display["sig"] = display["p"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    )

    col_labels = ["Factor", "Test", "Statistic", "df", "p-value", "η²", "Sig."]

    table_data = []
    for _, row in display.iterrows():
        table_data.append([
            row["factor"],
            row["test"],
            f"{row['stat']:.3f}"   if pd.notna(row["stat"])   else "—",
            str(row["df"]),
            f"{row['p']:.3f}"      if pd.notna(row["p"])      else "—",
            f"{row['eta_sq']:.3f}" if pd.notna(row["eta_sq"]) else "—",
            row["sig"],
        ])

    n_rows = len(table_data)
    fig, ax = plt.subplots(figsize=(11, max(2.5, n_rows * 0.85 + 1.2)))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.2)

    # Header
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    sig_col = len(col_labels) - 1
    for i, (_, row) in enumerate(display.iterrows(), start=1):
        is_overall = str(row["factor"]).lower() == "overall"
        is_sig     = row["p"] < 0.05
        sig_str    = row["sig"]

        base = "#edf2fb" if is_overall else ("#eaf4ea" if is_sig else ("white" if i % 2 else "#f7f7f7"))

        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_edgecolor("#e0e0e0")
            cell.set_linewidth(0.7)
            if j == sig_col:
                cell.set_facecolor(_SIG_COLORS.get(sig_str, "#eeeeee") + "44")
                cell.set_text_props(
                    color=_SIG_COLORS.get(sig_str, "#888888"),
                    fontweight="bold" if sig_str != "ns" else "normal",
                )
            else:
                cell.set_facecolor(base)
                cell.set_text_props(
                    color="#111111",
                    fontweight="bold" if is_overall else "normal",
                )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=14, color="#111111")
    _caption(fig, "Green row = significant (p < 0.05).  Blue row = Overall immersion.  "
             "η²: 0.01 small, 0.06 medium, 0.14+ large.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_anova_table(anova_df: pd.DataFrame) -> None:
    _anova_table_figure(
        anova_df,
        title="Statistical Test Results — IEQ-SF by LLM Temperature Condition",
        filename="anova_summary_table.png",
    )


def plot_overall_anova_table(overall_df: pd.DataFrame) -> None:
    display = overall_df.copy()
    display.insert(0, "factor", "Overall Immersion")
    _anova_table_figure(
        display,
        title="Statistical Test Result — Overall Immersion by LLM Temperature",
        filename="overall_immersion_anova_table.png",
    )


# ── Raw data table ─────────────────────────────────────────────────────────────

def plot_anova_raw_table(anova_raw: pd.DataFrame) -> None:
    FACTORS_LIST = ["Involvement", "RWD", "Challenge"]

    col_labels = [
        "Participant", "Temperature",
        "Involvement\n(Q1–Q4)", "Real World Dissociation\n(Q5–Q8)", "Challenge\n(Q9–Q11)",
    ]

    table_data = []
    row_meta   = []

    for temp in TEMPS:
        group = anova_raw[anova_raw["temperature"] == temp]
        for _, row in group.iterrows():
            table_data.append([
                str(row["player"]),
                TEMP_LABELS.get(str(row["temperature"]), str(row["temperature"])),
                f"{row['Involvement']:.2f}" if pd.notna(row["Involvement"]) else "—",
                f"{row['RWD']:.2f}"         if pd.notna(row["RWD"])         else "—",
                f"{row['Challenge']:.2f}"   if pd.notna(row["Challenge"])   else "—",
            ])
            row_meta.append(("data", temp))

        if len(group):
            means = group[FACTORS_LIST].mean()
            table_data.append([
                f"Group mean  (n = {len(group)})",
                TEMP_LABELS.get(temp, temp),
                f"{means['Involvement']:.2f}",
                f"{means['RWD']:.2f}",
                f"{means['Challenge']:.2f}",
            ])
            row_meta.append(("mean", temp))

    n_rows = len(table_data)
    fig, ax = plt.subplots(figsize=(12, max(5, n_rows * 0.48 + 1.5)))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.9)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    temp_bg   = {t: c + "25" for t, c in TEMP_COLORS.items()}
    temp_bold = {t: c + "65" for t, c in TEMP_COLORS.items()}

    for i, (kind, temp) in enumerate(row_meta, start=1):
        bg = temp_bold.get(temp, "#cccccc") if kind == "mean" else temp_bg.get(temp, "#f5f5f5")
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#e0e0e0")
            cell.set_linewidth(0.7)
            if kind == "mean":
                cell.set_text_props(fontweight="bold", color="#111111")

    ax.set_title("Raw ANOVA Input — IEQ-SF Factor Scores by Temperature Group",
                 fontsize=12, fontweight="bold", pad=12, color="#111111")
    _caption(fig, "Bold rows = group means.  Scores are Likert means (1–5).  "
             "RWD items Q5 and Q10 are reverse-scored before computing the mean.")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(PLOTS_DIR / "anova_raw_data_table.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Participant table ──────────────────────────────────────────────────────────

def plot_participant_table(scored: pd.DataFrame) -> None:
    df = scored.copy()
    df["temperature"] = pd.Categorical(df["temperature"], categories=TEMPS, ordered=True)
    df = df.sort_values(["temperature", "player"]).reset_index(drop=True)

    q_cols      = [f"Q{i}" for i in range(1, 12)]
    factor_cols = list(FACTORS.keys()) + ["Overall"]
    all_cols    = ["player", "temperature"] + q_cols + factor_cols
    col_labels  = ["Player", "Temp."] + q_cols + factor_cols

    rev_set        = {f"Q{i}" for i in REVERSED_ITEMS}
    factor_col_set = set(factor_cols)

    table_data = []
    for _, row in df[all_cols].iterrows():
        table_data.append([
            str(row["player"]),
            str(row["temperature"]),
            *[f"{row[c]:.0f}" if pd.notna(row[c]) else "—" for c in q_cols],
            *[f"{row[c]:.2f}" if pd.notna(row[c]) else "—" for c in factor_cols],
        ])

    n_cols = len(col_labels)
    n_rows = len(table_data)
    fig, ax = plt.subplots(figsize=(max(18, n_cols * 1.05), max(4, n_rows * 0.65 + 2.0)))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.85)

    HDR_BASE   = "#2c3e50"
    HDR_REV    = "#6c3483"
    HDR_FACTOR = "#1a5e36"

    for j, lbl in enumerate(col_labels):
        cell = tbl[0, j]
        cell.set_facecolor(
            HDR_REV    if lbl in rev_set else
            HDR_FACTOR if lbl in factor_col_set else
            HDR_BASE
        )
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    temp_bg   = {t: c + "20" for t, c in TEMP_COLORS.items()}
    factor_bg = "#e8f5ee"

    for i, (_, row) in enumerate(df[all_cols].iterrows(), start=1):
        t = str(row["temperature"])
        for j, lbl in enumerate(col_labels):
            cell = tbl[i, j]
            cell.set_facecolor(factor_bg if lbl in factor_col_set else temp_bg.get(t, "#fafafa"))
            cell.set_edgecolor("#e8e8e8")
            cell.set_linewidth(0.6)

    ax.set_title("Participant Scores — IEQ-SF (Q1–Q11)",
                 fontsize=11, fontweight="bold", pad=12, color="#111111")
    _caption(fig, "Purple header = reverse-scored item (Q5, Q10).  "
             "Green header = computed factor mean.  Involvement = Q1–Q4,  "
             "RWD = Q5–Q8,  Challenge = Q9–Q11,  Overall = Q1–Q11.")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(PLOTS_DIR / "participant_scores_table.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Item descriptives table ────────────────────────────────────────────────────

def plot_item_descriptives_table(scored: pd.DataFrame) -> None:
    factor_of = {f"Q{i}": factor for factor, items in FACTORS.items() for i in items}
    rev_set   = {f"Q{i}" for i in REVERSED_ITEMS}

    rows = []
    for q in range(1, 12):
        col    = f"Q{q}"
        factor = factor_of.get(col, "—")
        row    = {"Item": col, "Factor": factor, "Rev.": "Yes" if col in rev_set else ""}
        for temp in TEMPS:
            grp = scored.loc[scored["temperature"] == temp, col].dropna()
            row[f"{temp} M"]  = f"{grp.mean():.2f}" if len(grp) else "—"
            row[f"{temp} SD"] = f"{grp.std():.2f}"  if len(grp) > 1 else "—"
            row[f"{temp} n"]  = str(len(grp))
        rows.append(row)

    col_labels = ["Item", "Factor", "Rev."]
    for temp in TEMPS:
        col_labels += [f"{TEMP_LABELS[temp]}\nM", f"{TEMP_LABELS[temp]}\nSD", f"{TEMP_LABELS[temp]}\nn"]

    key_order = ["Item", "Factor", "Rev."]
    for temp in TEMPS:
        key_order += [f"{temp} M", f"{temp} SD", f"{temp} n"]
    table_data = [[r[k] for k in key_order] for r in rows]

    n_cols = len(col_labels)
    n_rows = len(table_data)
    fig, ax = plt.subplots(figsize=(max(15, n_cols * 1.35), max(4, n_rows * 0.62 + 2.0)))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.85)

    TEMP_HDR = {"T=0": "#2166ac", "T=0.35": "#b85c1e", "T=0.70": "#1a6b35"}
    BASE_HDR = "#2c3e50"

    header_col_map = ["Item", "Factor", "Rev."] + [t for t in TEMPS for _ in range(3)]
    for j, key in enumerate(header_col_map):
        cell = tbl[0, j]
        cell.set_facecolor(TEMP_HDR.get(key, BASE_HDR))
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    factor_colors = {"Involvement": "#dce9f5", "RWD": "#fdebd0", "Challenge": "#d5f0e0"}
    for i, row in enumerate(rows, start=1):
        bg = factor_colors.get(row["Factor"], "white") if i % 2 == 1 else "#f4f4f4"
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#e8e8e8")
            cell.set_linewidth(0.6)

    ax.set_title("Item-Level Descriptives — IEQ-SF (Q1–Q11) by Temperature",
                 fontsize=11, fontweight="bold", pad=12, color="#111111")
    _caption(fig, "M = mean,  SD = standard deviation.  "
             "Rev. = reverse-scored item (recoded as 6 − score before factor computation).")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(PLOTS_DIR / "item_descriptives_table.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Demographics table ─────────────────────────────────────────────────────────

def plot_demographics_table(demo_df: pd.DataFrame) -> None:
    col_map = {
        "player": "Player", "temperature": "Temp",
        "age": "Age", "gender": "Gender",
        "game_experience": "Game exp.\n(1–5)", "native_language": "Native lang.",
    }
    cols = [c for c in col_map if c in demo_df.columns]
    col_labels = [col_map[c] for c in cols]

    table_data = []
    for _, row in demo_df[cols].iterrows():
        table_data.append([
            str(row[c]) if pd.notna(row[c]) else "—" for c in cols
        ])

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.6), max(4, len(table_data) * 0.48 + 1.8)))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.9)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    temp_bg = {t: c + "25" for t, c in TEMP_COLORS.items()}
    for i, (_, row) in enumerate(demo_df[cols].iterrows(), start=1):
        bg = temp_bg.get(str(row["temperature"]), "#f5f5f5")
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#e0e0e0")
            cell.set_linewidth(0.7)

    ax.set_title("Participant Demographics", fontsize=11, fontweight="bold", pad=12, color="#111111")
    _caption(fig, "Game experience: 1 = no experience, 5 = very experienced.")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(PLOTS_DIR / "demographics_table.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Open answers + immersion summary ──────────────────────────────────────────

def plot_open_answers_table(scored: pd.DataFrame, open_df: pd.DataFrame) -> None:
    factor_cols = list(FACTORS.keys()) + ["Overall"]
    open_cols   = ["immersion_moment", "immersion_break", "memorable_moment"]
    col_display = {
        "player": "Player", "temperature": "Temp",
        "Involvement": "Inv.", "RWD": "RWD", "Challenge": "Chal.", "Overall": "Overall",
        "immersion_moment":  "Immersion moment (Q12)",
        "immersion_break":   "Immersion break (Q13)",
        "memorable_moment":  "Memorable moment (Q14)",
    }

    merged = (
        scored[["player", "temperature"] + factor_cols]
        .merge(open_df, on=["player", "temperature"], how="left")
    )
    merged["temperature"] = pd.Categorical(merged["temperature"], categories=TEMPS, ordered=True)
    merged = merged.sort_values(["temperature", "player"]).reset_index(drop=True)

    open_cols = [c for c in open_cols if c in merged.columns]
    all_cols   = ["player", "temperature"] + factor_cols + open_cols
    col_labels = [col_display.get(c, c) for c in all_cols]

    WRAP_WIDTH = 48

    def fmt(c, v):
        if not pd.notna(v):
            return "—"
        if c in factor_cols:
            return f"{v:.2f}"
        if c in open_cols:
            return textwrap.fill(str(v), WRAP_WIDTH)
        return str(v)

    table_data = [
        [fmt(c, row[c]) for c in all_cols]
        for _, row in merged[all_cols].iterrows()
    ]

    # Lines per row (drives row height)
    def row_lines(cells):
        return max(c.count("\n") + 1 for c in cells)

    line_counts = [row_lines(r) for r in table_data]

    FONT       = 8
    LINE_H     = 0.22   # inches per text line
    HDR_H      = 0.40   # header row height (inches)
    PAD        = 0.06   # per-row padding

    row_heights = [lc * LINE_H + PAD for lc in line_counts]
    total_h     = HDR_H + sum(row_heights)

    # Column width fractions (relative)
    raw_w  = [1.4, 0.85] + [0.65] * len(factor_cols) + [3.6] * len(open_cols)
    total_w = sum(raw_w)
    col_w  = [w / total_w for w in raw_w]

    fig_w = 24
    fig_h = total_h + 1.2   # title + caption
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="upper center",
                   bbox=[0, 0, 1, total_h / fig_h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT)

    # Set column widths
    for j, w in enumerate(col_w):
        for i in range(len(table_data) + 1):
            tbl[i, j].set_width(w)

    # Set row heights
    for j in range(len(all_cols)):
        tbl[0, j].set_height(HDR_H / fig_h)
    for i, rh in enumerate(row_heights, start=1):
        for j in range(len(all_cols)):
            tbl[i, j].set_height(rh / fig_h)

    # Left-align open-answer cells
    open_start = 2 + len(factor_cols)
    for i in range(1, len(table_data) + 1):
        for j in range(open_start, len(all_cols)):
            tbl[i, j]._loc = "left"

    # Header colours
    for j, c in enumerate(all_cols):
        cell = tbl[0, j]
        cell.set_facecolor(
            "#4a3070" if c in open_cols else
            "#1a5e36" if c in factor_cols else
            "#2c3e50"
        )
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    temp_bg = {t: c + "25" for t, c in TEMP_COLORS.items()}
    for i, (_, row) in enumerate(merged[all_cols].iterrows(), start=1):
        bg = temp_bg.get(str(row["temperature"]), "#f5f5f5")
        for j in range(len(all_cols)):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#e0e0e0")
            cell.set_linewidth(0.5)

    ax.set_title("Open Answers & Immersion Scores — IEQ-SF",
                 fontsize=12, fontweight="bold", pad=10, color="#111111")
    _caption(fig, "Green headers = computed factor scores (1–5).  "
             "Purple headers = open-ended responses.  Scores use reverse-coded items.",
             y=0.005)
    fig.savefig(PLOTS_DIR / "open_answers_table.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Unused detail figures (kept for reference, not called from main) ───────────

def _box_strip_panel(ax: plt.Axes, plot_data: pd.DataFrame, column: str) -> None:
    ax.set_facecolor("#f8f9fa")
    ax.yaxis.grid(True, color="white", linewidth=1.5, zorder=0)
    ax.set_axisbelow(True)

    for i, temp in enumerate(TEMPS):
        grp = plot_data.loc[plot_data["temperature"] == temp, column].dropna().values
        if len(grp) == 0:
            continue
        color = TEMP_COLORS[temp]
        dark  = _darken(color, 0.72)

        q1, med, q3 = np.percentile(grp, [25, 50, 75])
        iqr  = q3 - q1
        wlo  = max(grp.min(), q1 - 1.5 * iqr)
        whi  = min(grp.max(), q3 + 1.5 * iqr)

        ax.add_patch(mpatches.FancyBboxPatch(
            (i - 0.21, q1), 0.42, iqr,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=0.22,
            edgecolor=dark, linewidth=1.5, zorder=2,
        ))
        ax.plot([i - 0.21, i + 0.21], [med, med],
                color=dark, linewidth=2.5, solid_capstyle="round", zorder=4)
        for y_cap, y_box in [(wlo, q1), (whi, q3)]:
            ax.plot([i, i], [y_cap, y_box], color=dark, linewidth=1.2, zorder=2)
            ax.plot([i - 0.09, i + 0.09], [y_cap, y_cap], color=dark, linewidth=1.2, zorder=2)

    sns.stripplot(
        data=plot_data, x="temperature", y=column,
        hue="temperature", hue_order=TEMPS, palette=TEMP_COLORS,
        order=TEMPS, legend=False,
        size=7, jitter=0.10, alpha=0.88,
        linewidth=0.5, edgecolor="white",
        ax=ax, zorder=5,
    )

    for i, temp in enumerate(TEMPS):
        grp = plot_data.loc[plot_data["temperature"] == temp, column].dropna().values
        if len(grp):
            ax.scatter([i], [grp.mean()], marker="D", s=55, zorder=7,
                       facecolor="white", edgecolor=_darken(TEMP_COLORS[temp], 0.7),
                       linewidth=1.8)

    ax.set_ylim(0.6, 5.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.axhline(3, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
