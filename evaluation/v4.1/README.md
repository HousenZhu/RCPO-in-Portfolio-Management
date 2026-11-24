# V4.1 Evaluation

Each checkpoint selection contains one three-seed summary and three individual seed folders.
Each folder contains the Section 9 plots and the paper tables and learning figure.

- [Best return, seeds 0–2](best_return/average_seeds/paper_tables.md)
- [Feasibility-selected, seeds 0–2](best_feasible/average_seeds/paper_tables.md)

| Three-seed PDFs | Overall performance | Matched credit comparison | Training and deployment costs |
| --- | --- | --- | --- |
| Best return | [Table 1](best_return/average_seeds/table_1_main_results.pdf) | [Table 2](best_return/average_seeds/table_2_matched_credit.pdf) | [Table 3](best_return/average_seeds/table_3_compute.pdf) |
| Feasibility-selected | [Table 1](best_feasible/average_seeds/table_1_main_results.pdf) | [Table 2](best_feasible/average_seeds/table_2_matched_credit.pdf) | [Table 3](best_feasible/average_seeds/table_3_compute.pdf) |

The individual seed folders contain the same three PDF tables for their respective seed.

Both selections use the same 100 holdout markets. Figure 1 re-evaluates 31 snapshots per run on a fixed set of 20 markets. Section 9 validation history uses the original training logs. Per-step trajectory files are shared across folders.
