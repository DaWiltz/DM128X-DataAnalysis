<<<<<<< HEAD
from datahandling import (
=======
from __future__ import annotations

import pandas as pd

from calc import (
>>>>>>> f8e28d156339eebc160a182282d9347f88dbd83a
    OUT_DIR,
    PLOTS_DIR,
    build_conclusions,
    build_ieq_long,
    compute_descriptives,
    load_source_data,
    pivot_ieq,
    print_summary,
    run_factor_tests,
<<<<<<< HEAD
    score_ieq,
)
from graphing import (
    plot_anova_table,
    plot_factor_figures,
    plot_item_descriptives_table,
    plot_participant_table,
=======
    run_overall_test,
    score_ieq,
)
from graph import (
    plot_anova_raw_table,
    plot_anova_table,
    plot_item_descriptives_table,
    plot_participant_table,
    plot_summary_figure,
>>>>>>> f8e28d156339eebc160a182282d9347f88dbd83a
    plot_tukey_summary,
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
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
=======
    raw_df   = load_source_data()
    ieq_long = build_ieq_long(raw_df)
    wide     = pivot_ieq(ieq_long)
    scored   = score_ieq(wide)

    descriptives       = compute_descriptives(scored)
    anova_df, tukey_df = run_factor_tests(scored)
    overall_df         = run_overall_test(scored)

    # Merge Overall as the 4th row in the ANOVA table
    overall_row  = overall_df.copy()
    overall_row.insert(0, "factor", "Overall")
    anova_combined = pd.concat([anova_df, overall_row], ignore_index=True)

    anova_raw = (
        scored[["player", "temperature", "Involvement", "RWD", "Challenge"]]
        .sort_values(["temperature", "player"])
        .reset_index(drop=True)
    )

    anova_combined.to_csv(OUT_DIR / "anova_results.csv", index=False)
    tukey_df.to_csv(OUT_DIR / "tukey_results.csv", index=False)
    descriptives.to_csv(OUT_DIR / "descriptives.csv", index=False)
    anova_raw.to_csv(OUT_DIR / "anova_raw_data.csv", index=False)
    scored.to_csv(OUT_DIR / "participant_scores_q1_q11.csv", index=False)

    plot_summary_figure(scored)
    plot_tukey_summary(tukey_df)
    plot_anova_table(anova_combined)
    plot_anova_raw_table(anova_raw)
>>>>>>> f8e28d156339eebc160a182282d9347f88dbd83a
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
