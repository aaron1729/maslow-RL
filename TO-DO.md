after saturday 1/10:
- re-run the sweeps (on an A100, to do 6 simultaneously), saving all models (and all checkpoints) separately.
- check plots for sweeps.
- do evals on the resulting models, as well as on the linear run (at both checkpoints).
- run a sweep for the linear model as well, with the same values of beta (the multiplier).