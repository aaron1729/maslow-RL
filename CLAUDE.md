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

### 7. W&B Sweep Support

Added command-line arguments to enable hyperparameter sweeps:
```python
parser.add_argument("--k", type=float, help="Override gating steepness parameter")
parser.add_argument("--tau", type=float, help="Override gating threshold parameter")
parser.add_argument("--beta", type=float, help="Override correctness weight parameter")
```

This allows W&B to control parameters without modifying config files:
```bash
python train.py --config config.json --k 5 --tau 0.7 --beta 2.0
```

**Sweep configuration** (`sweep_gating.yaml`):
- Method: Random search (12 runs)
- Parameters: k=[2,3,5,10], tau=[0.3,0.5,0.7], beta=[1.0,2.0]
- Metric to optimize: `reward/accuracy`

**Parallel execution**:
- A10 (24GB): 2 agents → ~3-4 hours for 12 runs
- A100 (40GB): 6 agents → ~1-1.5 hours for 12 runs

### 8. Pacific Timezone Timestamps

W&B run names now include Pacific time timestamps for easier tracking:
```python
pacific_tz = pytz.timezone('America/Los_Angeles')
timestamp = datetime.now(pacific_tz).strftime("%Y-%m-%d, %H:%M")
run_name = f"{config['experiment']['name']} {timestamp}"
```

Example: "gsm8k-grpo-gated 2026-01-10, 13:45"

This helps distinguish runs when doing multiple experiments in a day.

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
data.py              - Dataset loading, chat template application
rewards.py           - Tier A/B computation, gating function, validation
train.py             - GRPO trainer setup, W&B logging, reward_fn, sweep support
eval.py              - Model evaluation, W&B histogram logging
sweep_gating.yaml    - W&B sweep config for hyperparameter search
config.json          - Gated run configuration
config_linear.json   - Linear baseline configuration
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

### Hyperparameter Sweeps
1. Initialize sweep: `wandb sweep sweep_gating.yaml` (returns sweep ID)
2. Launch agents: `wandb agent kernalabs/maslow-rl/<sweep-id>`
3. For parallel: Run multiple agents in background with nohup
4. Monitor: https://wandb.ai/kernalabs/maslow-rl/sweeps

**Resource planning**:
- 1 run ≈ 5.9 GB VRAM (0.5B model with LoRA)
- A10 (24GB): 2 agents max
- A100 (40GB): 6 agents max

### Model Storage
- **Code**: GitHub (aaron1729/maslow-RL)
- **Models**: Hugging Face (aaron1729/maslow-rl-gsm8k-gated)
- **Metrics**: W&B (kernalabs/maslow-rl)
- **Sweeps**: W&B (kernalabs/maslow-rl/sweeps)

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

## Critical Bugs Fixed (2026-01-10)

### 1. Model Overwriting in Sweeps

**Problem**: All sweep runs saved to the same output directory (`outputs/gsm8k-grpo-gated/`), causing each run to overwrite previous models. Lost 17 of 18 trained models from initial hyperparameter sweep.

**Root Cause**: Output directory was based only on `experiment.name` from config, not unique per run.

**Fix**: Append W&B run ID and Pacific timestamp to output directory:
```python
base_name = experiment_config["name"]
if wandb.run is not None:
    timestamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
    base_name = f"{base_name}-{wandb.run.id}-{timestamp}"
output_dir = Path(experiment_config["output_dir"]) / base_name
```

Example: `outputs/gsm8k-grpo-gated-abc123-20260110-153045/`

### 2. Sample Completions Missing Metadata

**Problem**: Sample completions logged without run identification or hyperparameters. 2736 completions from 18 runs were mixed together with no way to separate them.

**Fix**: Added metadata to each log entry:
```python
log_entry = {
    "wandb_run_id": wandb.run.id,
    "wandb_run_name": wandb.run.name,
    "run_type": run_type,  # "gated" or "linear"
    "k": config["rewards"]["gating"]["k"],
    "tau": config["rewards"]["gating"]["tau"],
    "beta": config["rewards"]["gating"]["beta"],
    "step": reward_fn.call_count,
    "question": questions[i],  # Original GSM8K question
    "target": target_ints[i],
    "completion": completion_texts[i],
    "r_a": round(infos[i]["r_a"], 1),
    "r_b": round(infos[i]["r_b"], 1),
    "gate_b": round(infos[i]["gate_b"], 4),
    "reward": round(rewards[i], 4)
}
```

Now each completion includes:
- Run identification (ID, name)
- Hyperparameters (k, tau, beta)
- Original question text (for validation)
- Properly rounded floats (no more 0.9999999999999999)

### 3. W&B Sweep Infinite Loops

**Problem**: W&B sweeps with `count: 12` and multiple agents don't auto-stop. Agents keep polling for work after 12 runs complete, leading to duplicate configs.

**Symptoms**:
- 18 runs completed instead of 12
- 2 duplicate configurations (k=2, tau=0.7, beta=2 and k=5, tau=0.3, beta=2 ran twice)
- Random sampling can hit the same config multiple times

**Solution**: Manually kill agents when desired run count is reached:
```bash
pkill -f "wandb agent"
```

Or monitor sweep and mark as finished in W&B UI when count is reached.

### 4. W&B Histogram Display
Tried multiple approaches to log distributions:
- `wandb.Histogram()` - Rendered as constant functions
- `wandb.plot.histogram()` - Created 1-2 giant bins hiding all variation

**Solution**: Use matplotlib to generate histograms, upload as images:
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... create histograms with proper bins ...
wandb.log({"eval/distributions": wandb.Image(fig)})
```

Also save to local file for offline analysis.

### 5. Lambda Labs Instance Persistence
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

## Rescued Model

After the sweep overwriting disaster, one complete model survived:

**HuggingFace**: `aaron1729/maslow-rl-gsm8k-gated`
- **Config**: k=5, tau=0.3, beta=2
- **Checkpoints**: 250, 500, 750, 1000 steps + final_model
- **Directory structure**: `k=5_tau=0.3_beta=2/checkpoint-{250,500,750,1000,final_model}/`

This model represents 1 of the 18 sweep runs. The other 17 configurations were lost to overwriting.

**Rescued data**:
- `sample_completions_MIXED_RUNS_2026-01-10.jsonl` - 2736 completions from all 18 runs
  - Cannot separate by hyperparameters (logged before metadata fix)
  - Useful for aggregate analysis of learning trajectory
  - Shows progression from step 10 to step 1000

## Lessons Learned

1. **Start with low k for gating** - Steep sigmoids create hard thresholds that hurt learning
2. **Fix logging bugs early** - We lost visibility into half the training run
3. **Validate rewards** - Step quality check prevents gaming
4. **Use Hugging Face for model storage** - Don't rely on cloud instance persistence
5. **Temperature matters** - Higher temp for training, lower for eval
6. **Smaller models iterate faster** - 0.5B was perfect for this experiment
7. **W&B sweeps accelerate research** - 12 hyperparameter combinations in 1.5 hours with A100
8. **Parallel agents save time** - 6 agents on A100 → 4x faster than sequential runs
9. **Command-line overrides beat config editing** - Enables sweeps without code changes
10. **Timestamp run names** - Pacific time timestamps make it easy to correlate runs with notes
11. **CRITICAL: Unique output directories for sweeps** - Include run ID and timestamp in paths
12. **Log metadata with everything** - Run IDs, hyperparameters, original questions
13. **Monitor sweeps actively** - W&B agents don't auto-stop at count limit
14. **Test with small runs first** - Catch bugs before committing to long sweeps
15. **Checkpoint early and often** - Cloud instances can fail unexpectedly

## References

- GRPO paper: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- TRL library: [huggingface/trl](https://github.com/huggingface/trl)
- GSM8K dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- Qwen2 models: [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
