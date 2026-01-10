# Evaluation Results

**Model:** aaron1729/maslow-rl-gsm8k-gated  
**Timestamp:** 2026-01-10 19:46:54 UTC  
**W&B Run:** https://wandb.ai/kernalabs/maslow-rl/runs/1mbx52r4  
**Dataset:** GSM8K test set (100 samples)

## Results

- **Accuracy:** 37.0%
- **R_A (Structure) Mean:** 0.986 (98.6%)
- **R_B (Correctness) Mean:** 0.370 (37.0%)
- **Gate_B Mean:** 0.809
- **Total Reward Mean:** 1.286
- **JSON Parse Rate:** 99%
- **Valid Schema Rate:** 98%
- **R_A Perfect Rate:** 98%

## Interpretation

The model has **mastered the structural requirements** (98.6% R_A), producing valid JSON with proper step formatting in nearly all cases. However, **mathematical correctness** is lower at 37%, indicating the model can format reasoning steps correctly but struggles with accurate problem-solving.

The hierarchical gating approach successfully taught structure first (R_A ≈ 1.0), with the gate now mostly open (0.81) to allow correctness rewards to influence training.

## Files

- `eval_distributions_*.png` - Histogram plots of reward distributions
- `eval_results_*.json` - Full results including all samples and metrics
- `eval_run_*.log` - Console output from evaluation run
