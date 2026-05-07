from datahandling import (
    OUT_DIR,
    PLOTS_DIR,
    build_conclusions,
    build_ieq_long,
    compute_descriptives,
    load_source_data,
    pivot_ieq,
    print_summary,
    run_factor_tests,
    score_ieq,
)
from graphing import (
    plot_anova_table,
    plot_factor_figures,
    plot_item_descriptives_table,
    plot_participant_table,
    plot_tukey_summary,
)


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
    plot_anova_table(anova_df)
    plot_participant_table(scored)
    plot_item_descriptives_table(scored)

    conclusions = build_conclusions(anova_df, tukey_df)
    (OUT_DIR / "conclusions.txt").write_text(conclusions, encoding="utf-8")

    print_summary(descriptives, anova_df)
    print(f"\n{conclusions}")
    print(f"\nSaved CSV outputs to: {OUT_DIR.resolve()}")
    print(f"Saved plots to: {PLOTS_DIR.resolve()}")
    print(f"Participants processed: {scored['player'].nunique()}")


if __name__ == "__main__":
    main()
