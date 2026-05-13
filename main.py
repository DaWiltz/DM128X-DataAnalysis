from __future__ import annotations

import pandas as pd

from calc import (
    OUT_DIR,
    PLOTS_DIR,
    build_conclusions,
    build_ieq_long,
    compute_descriptives,
    extract_demographics,
    extract_open_answers,
    load_source_data,
    pivot_ieq,
    print_summary,
    run_factor_tests,
    run_overall_test,
    score_ieq,
)
from graph import (
    plot_anova_table,
    plot_demographics_table,
    plot_item_descriptives_table,
    plot_open_answers_table,
    plot_participant_table,
    plot_summary_figure,
    plot_tukey_summary,
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_source_data()

    # Demographics and open answers (from consent form)
    demo_df = extract_demographics(raw_df)
    open_df = extract_open_answers(raw_df)

    # IEQ pipeline
    scored = score_ieq(pivot_ieq(build_ieq_long(raw_df)))

    # Stats
    descriptives       = compute_descriptives(scored)
    anova_df, tukey_df = run_factor_tests(scored)
    overall_df         = run_overall_test(scored)

    overall_row = overall_df.copy()
    overall_row.insert(0, "factor", "Overall")
    anova_combined = pd.concat([anova_df, overall_row], ignore_index=True)

    # Save CSVs
    scored.to_csv(OUT_DIR / "participant_scores.csv", index=False)
    demo_df.to_csv(OUT_DIR / "demographics.csv", index=False)
    open_df.to_csv(OUT_DIR / "open_answers.csv", index=False)
    descriptives.to_csv(OUT_DIR / "descriptives.csv", index=False)
    anova_combined.to_csv(OUT_DIR / "anova_results.csv", index=False)
    tukey_df.to_csv(OUT_DIR / "tukey_results.csv", index=False)

    # Plots
    plot_summary_figure(scored)
    plot_demographics_table(demo_df)
    plot_open_answers_table(scored, open_df)
    plot_tukey_summary(tukey_df)
    plot_anova_table(anova_combined)
    plot_participant_table(scored)
    plot_item_descriptives_table(scored)

    # Conclusions
    conclusions = build_conclusions(anova_df, tukey_df)
    (OUT_DIR / "conclusions.txt").write_text(conclusions, encoding="utf-8")

    print_summary(descriptives, anova_df)
    print(f"\n{conclusions}")
    print(f"\nOutputs → {OUT_DIR.resolve()}")
    print(f"Plots   → {PLOTS_DIR.resolve()}")
    print(f"N       = {scored['player'].nunique()} participants")


if __name__ == "__main__":
    main()
