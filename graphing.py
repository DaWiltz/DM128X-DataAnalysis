from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datahandling import (
    FACTORS,
    PLOTS_DIR,
    REVERSED_ITEMS,
    TEMP_COLORS,
    TEMP_LABELS,
    TEMPS,
)


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
            if not np.isfinite(top):
                continue
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


def plot_participant_table(scored: pd.DataFrame) -> None:
    df = scored.copy()
    df["temperature"] = pd.Categorical(df["temperature"], categories=TEMPS, ordered=True)
    df = df.sort_values(["temperature", "player"]).reset_index(drop=True)

    q_cols = [f"Q{i}" for i in range(1, 12)]
    factor_cols = list(FACTORS.keys()) + ["Overall"]
    all_cols = ["player", "temperature"] + q_cols + factor_cols

    col_labels = ["Player", "Temp"] + q_cols + factor_cols

    rev_set = {f"Q{i}" for i in REVERSED_ITEMS}
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
    fig, ax = plt.subplots(figsize=(max(18, n_cols * 1.1), max(3, n_rows * 0.7 + 2.5)))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.9)

    header_color = "#4C72B0"
    rev_header   = "#8B6090"
    factor_header = "#2E6B4F"

    for j, lbl in enumerate(col_labels):
        col = lbl
        if col in rev_set:
            hc = rev_header
        elif col in factor_col_set:
            hc = factor_header
        else:
            hc = header_color
        tbl[0, j].set_facecolor(hc)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    temp_bg = {t: c + "22" for t, c in TEMP_COLORS.items()}
    factor_bg = "#e8f5ee"

    for i, (_, row) in enumerate(df[all_cols].iterrows(), start=1):
        t = str(row["temperature"])
        base = temp_bg.get(t, "#fafafa")
        for j, lbl in enumerate(col_labels):
            bg = factor_bg if lbl in factor_col_set else base
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        "Participant Scores — IEQ-SF\n"
        f"(purple header = reverse-scored item; green header = computed factor/overall)",
        fontsize=12, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "participant_scores_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_item_descriptives_table(scored: pd.DataFrame) -> None:
    factor_of = {f"Q{i}": factor for factor, items in FACTORS.items() for i in items}
    rev_set = {f"Q{i}" for i in REVERSED_ITEMS}

    rows = []
    for q in range(1, 12):
        col = f"Q{q}"
        factor = factor_of.get(col, "—")
        rev = "Yes" if col in rev_set else ""
        row = {"Item": col, "Factor": factor, "Reversed": rev}
        for temp in TEMPS:
            grp = scored.loc[scored["temperature"] == temp, col].dropna()
            if len(grp):
                row[f"{temp} mean"] = f"{grp.mean():.2f}"
                row[f"{temp} SD"]   = f"{grp.std():.2f}" if len(grp) > 1 else "—"
                row[f"{temp} n"]    = str(len(grp))
            else:
                row[f"{temp} mean"] = "—"
                row[f"{temp} SD"]   = "—"
                row[f"{temp} n"]    = "0"
        rows.append(row)

    col_labels = ["Item", "Factor", "Rev."]
    for temp in TEMPS:
        col_labels += [f"{TEMP_LABELS[temp]}\nmean", f"{TEMP_LABELS[temp]}\nSD", f"{TEMP_LABELS[temp]}\nn"]

    key_order = ["Item", "Factor", "Reversed"]
    for temp in TEMPS:
        key_order += [f"{temp} mean", f"{temp} SD", f"{temp} n"]
    table_data = [[r[k] for k in key_order] for r in rows]

    n_cols = len(col_labels)
    n_rows = len(table_data)
    fig, ax = plt.subplots(figsize=(max(16, n_cols * 1.4), max(4, n_rows * 0.65 + 2.5)))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.9)

    temp_header_colors = {"T=0": "#4C72B0", "T=0.35": "#DD8452", "T=0.70": "#55A868"}
    base_header = "#555555"

    header_col_map = ["Item", "Factor", "Reversed"] + [
        t for t in TEMPS for _ in range(3)
    ]
    for j, key in enumerate(header_col_map):
        hc = temp_header_colors.get(key, base_header)
        tbl[0, j].set_facecolor(hc)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    factor_colors = {"Involvement": "#dce9f5", "RWD": "#fdebd0", "Challenge": "#d5f0e0"}
    for i, row in enumerate(rows, start=1):
        bg = factor_colors.get(row["Factor"], "white") if i % 2 == 1 else "#f7f7f7"
        for j in range(n_cols):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        "Item-Level Descriptives — IEQ-SF (Q1–Q11) by Temperature\n"
        "(Rev. = reverse-scored before factor computation)",
        fontsize=12, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "item_descriptives_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_anova_table(anova_df: pd.DataFrame) -> None:
    display = anova_df.copy()
    display["test"] = display["test"].str.upper().replace("KRUSKAL", "Kruskal-Wallis").replace("ANOVA", "One-way ANOVA")
    display["sig"] = display["p"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    )

    col_keys   = ["factor", "test", "stat", "df", "p",       "eta_sq", "sig"]
    col_labels = ["Factor", "Test", "Statistic", "df", "p-value", "η²",  "Sig."]

    table_data = []
    for _, row in display.iterrows():
        table_data.append([
            row["factor"],
            row["test"],
            f"{row['stat']:.3f}" if pd.notna(row["stat"]) else "—",
            str(row["df"]),
            f"{row['p']:.3f}"    if pd.notna(row["p"])    else "—",
            f"{row['eta_sq']:.3f}" if pd.notna(row["eta_sq"]) else "—",
            row["sig"],
        ])

    fig, ax = plt.subplots(figsize=(11, max(2.5, len(display) * 0.8 + 1.8)))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    header_color = "#4C72B0"
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(header_color)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, (_, row) in enumerate(display.iterrows(), start=1):
        row_color = "#e8f5e9" if row["p"] < 0.05 else "#fafafa" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(row_color)

    ax.set_title(
        "Statistical Test Results (ANOVA / Kruskal-Wallis)\n"
        "Significance: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant",
        fontsize=12, fontweight="bold", pad=16,
    )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "anova_summary_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
