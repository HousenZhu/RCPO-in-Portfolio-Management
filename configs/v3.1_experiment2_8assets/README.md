# V3.1 Experiment 2

V3.1 keeps the V2.6 eight-risky-asset environment and autoregressive Dirichlet
simplex policy. Its experimental change is the RCPO dual signal: lambda can be
updated from the rolling 80th percentile of complete-episode cost-minus-alpha
gaps instead of the mean over all rollout steps.

- Pilot: compare `lambda_lr_up` 0.00025 and 0.00040 over 14,000 updates.
- Final: run the selected quantile setting over 40,000 updates and compare it
  with the V2.6-style mean-step control.
- The rolling window contains 64 complete episodes and activates after 16.
- Validation runs every 100 updates; live validation PNG rewriting is disabled.
- Final evaluation artifacts are still generated when training completes.
