# Maslow RL - Technical Notes

Implementation notes and design decisions for the hierarchical reward shaping experiment.

## Implementation Overview

This project trains a small LLM (Qwen2-0.5B) on GSM8K math problems using GRPO (Group Relative Policy Optimization) with a hierarchical reward function that gates correctness rewards behind structural quality.

## Key Design Decisions

### 1. Gating Parameters (k=3, tau=0.5)

**Why k=3 instead of k=20?**
- Lower steepness creates a smoother gradient
- k=20 was too steep - effectively a hard threshold
- k=3 allows the model to benefit from partial structure improvements
- Gate transitions gradually from 0.18 (R_A=0) to 0.82 (R_A=1.0)

**Why tau=0.5 instead of 0.85?**
- More forgiving threshold - model doesn't need near-perfect structure before getting correctness rewards
- 0.5 means "decent structure" triggers partial correctness rewards
- Empirically, this led to better exploration and less reward hacking

### 2. Step Quality Validation

Added validation to prevent trivial step generation:
- **Length**: 15-500 characters per step
- **Content**: Must contain numbers OR math operators (+, -, *, /, =)
- **Minimum steps**: At least 2 steps required

This prevents the model from gaming the system with empty or meaningless steps like:
```json
{
  "step_1": "x",
  "step_2": "y",
  "answer": 42
}
```

### 3. Temperature and Exploration

**Training temperature: 1.0** (increased from 0.7)
- Higher temperature improves exploration during training
- GRPO benefits from diverse generations per prompt
- 8 generations per problem with temp=1.0 provides good variety

**Eval temperature: 0.1**
- Low temperature for deterministic evaluation
- Shows "typical" model behavior, not best-case sampling

### 4. W&B Logging Bug Fix

**Problem**: GRPO calls `reward_fn` 8 times per training step (once per problem in batch). Original implementation logged each call separately, so W&B only showed metrics from the last call.

**Solution**:
```python
reward_fn_state = {
    'step_metrics': [],
    'last_logged_step': -1,
    'batch_size': batch_size
}

# Accumulate metrics across all calls in a step
reward_fn_state['step_metrics'].append(call_metrics)

# Log to W&B only after all batch_size calls complete
if reward_fn.call_count % batch_size == 0:
    # Aggregate all r_a, r_b, gate_b values
    wandb.log({...})
    reward_fn_state['step_metrics'] = []
```

This ensures W&B shows the true mean across all 64 completions (8 problems × 8 generations).

### 5. KL Divergence Coefficient

**kl_coef: 0.01** (reduced from 0.05)
- GRPO includes KL penalty to prevent model from diverging too far from base
- Lower coefficient allows more exploration
- Still prevents catastrophic forgetting

### 6. Chat Template Handling

GRPO expects prompts as message lists:
```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Problem: ..."}
]
```

eval.py needed to handle this format:
```python
if isinstance(prompt, list):
    prompt_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
```

## Architecture Choices

### Why Qwen2-0.5B?
- Small enough for fast iteration ($0.75/hr on A10)
- Large enough to learn reasoning patterns
- Good tokenizer and chat template support
- Strong base performance on math

### Why LoRA instead of full fine-tuning?
- Much faster (8.3MB vs 1GB+ checkpoints)
- Lower memory requirements
- Easier to experiment with multiple runs
- Can merge adapters for deployment

### Why GRPO instead of PPO/DPO?
- Simpler than PPO (no separate critic model)
- More suitable than DPO for online RL with custom rewards
- Designed specifically for reasoning tasks
- Built into TRL with good defaults

## File Organization

```
data.py         - Dataset loading, chat template application
rewards.py      - Tier A/B computation, gating function, validation
train.py        - GRPO trainer setup, W&B logging, reward_fn
eval.py         - Model evaluation, W&B histogram logging
lambda_deploy.py - Lambda Labs instance management
```

## Workflow

### Local Development
1. Make code changes locally
2. Commit to git: `git commit -m "..."`
3. Push to GitHub: `git push origin main`

### Lambda Training
1. SSH into instance
2. Pull latest code: `git pull origin main`
3. Activate venv: `source venv/bin/activate`
4. Run training: `python train.py --config config.json`

### Model Storage
- **Code**: GitHub (aaron1729/maslow-RL)
- **Models**: Hugging Face (aaron1729/maslow-rl-gsm8k-gated)
- **Metrics**: W&B (kernalabs/maslow-rl)

## Results Analysis

### Training Dynamics (500 steps)

**Phase 1 (Steps 0-70): Structure Learning**
- R_A rises from ~0.3 to ~0.95
- R_B remains low (~0.1-0.3)
- Gate_B rises with R_A
- Model learns JSON format, step structure, numeric answers

**Phase 2 (Steps 70-500): Correctness Learning**
- R_A plateaus at ~0.98
- R_B improves but unstable (0.0 to 0.6, bouncing)
- Gate_B ~0.8 (mostly open)
- Model attempts correct arithmetic, sometimes succeeds

**Why R_B bounces?**
- GRPO samples 8 problems per step - some easier than others
- Math reasoning is harder than structure
- 500 steps might not be enough for full convergence
- Natural variance in generation quality

### Evaluation Results

- **JSON Parse Rate**: 99% - Nearly perfect format
- **Valid Schema Rate**: 98% - Proper step structure
- **R_A Mean**: 98.6% - Structure mastered
- **Accuracy (R_B)**: 42-47% - Room for improvement

**Interpretation**: Hierarchical approach successfully taught structure first. Correctness needs more training or stronger base model.

## Future Improvements

### Short Term
1. **More training steps**: Run 1000-2000 steps
2. **Larger batch size**: More problems per step for stable R_B
3. **Curriculum learning**: Start with easier problems
4. **Linear baseline**: Compare against non-gated rewards

### Medium Term
1. **Tier C (Reasoning Quality)**: Add LLM-as-judge for step quality
2. **Larger model**: Scale to Qwen2-1.5B or 7B
3. **Better prompts**: Provide more examples, clearer instructions
4. **Data augmentation**: Generate similar problems for more training data

### Research Directions
1. **Adaptive gating**: Learn k and tau during training
2. **Multi-tier hierarchy**: Add more intermediate rewards
3. **Transfer to other domains**: Code, planning, scientific reasoning
4. **Theoretical analysis**: Prove convergence properties of gated rewards

## Known Issues

### W&B Histogram Display
Tried multiple approaches to log distributions:
- `wandb.Histogram()` - Rendered as constant functions
- `wandb.plot.histogram()` - Still not displaying correctly

**Workaround**: All eval data saved to `eval_results.json` with full per-sample breakdowns. Can plot distributions locally with matplotlib.

### Lambda Labs Instance Persistence
- No "pause" option - only terminate or keep running
- Terminating destroys local storage
- Must use Hugging Face Hub for model persistence

## Environment Setup

### Lambda Labs
```bash
# On fresh instance
git clone https://github.com/aaron1729/maslow-RL.git
cd maslow-RL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy .env file (don't commit!)
scp local:.env ubuntu@instance:~/maslow-rl/

# Run training
python train.py --config config.json
```

### Local Testing
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test components
python data.py
python rewards.py

# Quick test run (50 steps)
python train.py --config config_test.json
```

## Lessons Learned

1. **Start with low k for gating** - Steep sigmoids create hard thresholds that hurt learning
2. **Fix logging bugs early** - We lost visibility into half the training run
3. **Validate rewards** - Step quality check prevents gaming
4. **Use Hugging Face for model storage** - Don't rely on cloud instance persistence
5. **Temperature matters** - Higher temp for training, lower for eval
6. **Smaller models iterate faster** - 0.5B was perfect for this experiment

## References

- GRPO paper: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- TRL library: [huggingface/trl](https://github.com/huggingface/trl)
- GSM8K dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- Qwen2 models: [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
