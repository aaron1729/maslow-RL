# Maslow RL: Hierarchical Reward Shaping for Math Reasoning

This repository implements a reinforcement learning experiment demonstrating that **hierarchically gated rewards** produce better learning dynamics than linear reward combination for training language models on math reasoning tasks.

## Core Hypothesis

Models learn more efficiently when rewards follow a hierarchy - like Maslow's hierarchy of needs. Lower-tier requirements must be satisfied before higher-tier rewards activate:

1. **Tier A (Structure)**: Valid JSON with proper schema
2. **Tier B (Correctness)**: Numerically correct answer
3. **Tier C (Reasoning)**: High-quality step-by-step reasoning (optional)

## Experiment Design

We compare two training runs on GSM8K:

### Run A: Linear Baseline
```
R = R_A + β·R_B
```
All rewards contribute equally from the start.

### Run B: Gated (Maslow)
```
R = R_A + β·σ(k(R_A - τ))·R_B
```
Where:
- σ(x) = sigmoid function
- k = 20 (steepness)
- τ = 0.85 (threshold)
- β = 1.0 (weight)

The sigmoid gate means **R_B only contributes when R_A ≥ 0.85**, forcing the model to master structure before being rewarded for correctness.

### Expected Outcome

The gated run should show a "staircase" learning pattern:
- **Phase 1**: R_A rises quickly (learning JSON structure)
- **Phase 2**: R_B rises after structure is solid (learning correctness)
- **Result**: Clearer separation of skills, more stable training

## Reward Functions

### Tier A: Structure Reward (R_A ∈ [0,1])

| Component | Weight | Description |
|-----------|--------|-------------|
| Parsable JSON | 0.3 | Valid JSON after fence stripping |
| Valid Schema | 0.4 | Has `step_1`, `step_2`, ..., `answer` keys in sequence |
| Numeric Answer | 0.2 | `answer` field is parseable as number |
| JSON-only | 0.1 | No preamble or trailing text |

Example valid output:
```json
{
  "step_1": "First, calculate the total items...",
  "step_2": "Then multiply by the unit cost...",
  "step_3": "Finally, add the shipping fee...",
  "answer": 42
}
```

### Tier B: Correctness Reward (R_B ∈ {0,1})

- **1.0** if `|predicted_answer - target| < 0.001`
- **0.0** otherwise

### Gating Function

```python
gate_b = sigmoid(k * (R_A - tau))
       = 1 / (1 + exp(-20 * (R_A - 0.85)))
```

When R_A = 0.85, gate_b = 0.5 (half contribution)
When R_A = 1.0, gate_b ≈ 0.95 (nearly full contribution)
When R_A = 0.70, gate_b ≈ 0.05 (minimal contribution)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import transformers; import trl; print('Setup complete!')"
```

## Dataset

- **Source**: GSM8K (Grade School Math 8K)
- **Train**: 500 examples sampled with seed=42
- **Eval**: 100 examples (disjoint from train)
- **Target Extraction**: Extracts integer after `####` separator

## Configuration

All hyperparameters are in JSON config files:

- `config.json` - Gated (Maslow) run
- `config_linear.json` - Linear baseline run

### Key Config Sections

**Model & LoRA**:
```json
{
  "model": {
    "name": "Qwen/Qwen2.5-0.5B-Instruct",
    "dtype": "bf16"
  },
  "lora": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  }
}
```

**GRPO Settings**:
```json
{
  "grpo": {
    "num_generations_per_prompt": 8,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "kl_coef": 0.05
  }
}
```

**Reward Gating**:
```json
{
  "rewards": {
    "gating": {
      "k": 20,
      "tau": 0.85,
      "beta": 1.0
    }
  }
}
```

Edit these files to adjust hyperparameters without modifying code.

## Usage

### 1. Test Components

Test data loading:
```bash
python data.py
```

Test reward computation:
```bash
python rewards.py
```

### 2. Run Training

**Gated (Maslow) run**:
```bash
python train.py --config config.json
```

**Linear baseline**:
```bash
python train.py --config config_linear.json
```

Or override run type:
```bash
python train.py --config config.json --run-type linear
```

### 3. Evaluate Model

```bash
python eval.py \
  --config config.json \
  --model-path ./outputs/gsm8k-grpo-gated/final_model \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --num-samples 100
```

Results saved to `eval_results.json` with:
- Aggregate metrics (accuracy, R_A, R_B, etc.)
- Sample outputs for inspection

## Monitoring Training

### Without W&B (stdout logs)

Training logs will show:
```
Step 100: reward_total=0.85, r_a=0.82, r_b=0.15, gate_b=0.23
Step 200: reward_total=1.02, r_a=0.95, r_b=0.45, gate_b=0.87
```

### With W&B

Enable in config:
```json
{
  "experiment": {
    "wandb_enabled": true,
    "wandb_project": "maslow-rl"
  }
}
```

Then:
```bash
wandb login
python train.py --config config.json
```

Tracked metrics:
- `reward_total_mean`
- `r_a_mean`
- `r_b_mean`
- `gate_b_mean` (gated run only)
- `accuracy`
- `json_parse_rate`

## Expected Results

### Gated Run Learning Curve

```
R_A: ▁▂▄▆█████████  (rises first)
R_B: ▁▁▁▁▂▄▆█████  (rises later)
Gate: ▁▁▃▅███████  (tracks R_A)
```

### Linear Run Learning Curve

```
R_A: ▁▂▄▆████████  (gradual)
R_B: ▁▂▃▅▆███████  (gradual)
```

The gated run should show:
1. Clearer phase separation
2. Higher R_A achieved earlier
3. More stable R_B improvement after gating opens

## File Structure

```
maslow-RL/
├── config.json              # Gated run config
├── config_linear.json       # Linear run config
├── data.py                  # Dataset loading & preprocessing
├── rewards.py               # Reward computation (Tier A/B)
├── train.py                 # GRPO training script
├── eval.py                  # Evaluation script
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── outputs/                # Training outputs (created)
    ├── gsm8k-grpo-gated/
    │   ├── checkpoint-500/
    │   ├── final_model/
    │   └── config.json
    └── gsm8k-grpo-linear/
        └── ...
```

## Budget & Compute

- **Model**: Qwen2.5-0.5B (small, efficient)
- **GPU**: Single A10 or L4 (16-24GB VRAM)
- **Training**: ~1000 steps, ~30-60 minutes
- **Cost**: ~$5-10 on cloud GPU providers

Example providers:
- Lambda Labs: A10 @ $0.60/hr
- RunPod: RTX 4090 @ $0.39/hr
- Vast.ai: Various GPUs @ $0.20-0.60/hr

## Troubleshooting

**Out of memory?**
- Reduce `per_device_train_batch_size` in config
- Reduce `num_generations_per_prompt`
- Use `gradient_accumulation_steps` to maintain effective batch size

**Training unstable?**
- Lower learning rate (try 5e-5)
- Increase warmup steps
- Check if gating threshold is too strict (try tau=0.80)

**Model not learning structure?**
- Inspect samples with `eval.py`
- Check if prompt template is clear
- Try increasing Tier A weights

**GRPO not available?**
- Update TRL: `pip install --upgrade trl`
- Check TRL version: `python -c "import trl; print(trl.__version__)"`
- Required: trl >= 0.8.0

## Citation

If you use this code or extend the experiment:

```bibtex
@misc{maslow-rl-2026,
  title={Hierarchical Reward Shaping for Mathematical Reasoning},
  author={Maslow RL Experiment},
  year={2026},
  url={https://github.com/yourusername/maslow-rl}
}
```

## Future Directions

1. **Tier C (Reasoning Quality)**: Add external judge for reasoning evaluation
2. **Adaptive Gating**: Learn k and tau during training
3. **Multi-Stage Training**: Curriculum with increasing difficulty
4. **Larger Models**: Scale to 1B-7B parameter models
5. **Other Domains**: Apply to code generation, planning, etc.

## License

MIT License - See LICENSE file for details
