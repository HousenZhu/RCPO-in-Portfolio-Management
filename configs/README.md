# Configuration Layout

- `default.yaml`: backward-compatible project defaults and legacy interactive experiments.
- `experiment1_current/`: preserved five-risky-asset experiment documentation.
- `experiment2_8assets/`: preserved V2.5 eight-risky-asset configurations and environment definition.
- `v2.6_experiment2_8assets/`: authoritative Phase 1 V2.6 six-run configurations.
- Root-level `simplex_*.yaml` and `rcpo_*.yaml`: legacy full-file configs retained so historical commands do not break.

New V2.6 training should use only the files under `v2.6_experiment2_8assets/`.

