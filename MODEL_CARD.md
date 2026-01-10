---
base_model: Qwen/Qwen2-0.5B-Instruct
library_name: transformers
model_name: maslow-rl-gsm8k-gated
tags:
- trl
- grpo
- reinforcement-learning
- math-reasoning
- hierarchical-rewards
license: apache-2.0
---

# Maslow RL: Hierarchical Reward Shaping for Math Reasoning

This model demonstrates **hierarchical reward shaping** (Maslow RL) - a technique that teaches LLMs to first master output structure before optimizing for correctness.

Fine-tuned from [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on GSM8K using GRPO with a gated reward function.

## Training Results

**500 training steps:**
- Structure quality (r_a): **98.6%** - Model learned to produce valid JSON with reasoning steps
- Test accuracy (r_b): **42-47%** - Correctness improving
- Successfully validates hierarchical approach: structure learned first, then correctness

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/kernalabs/maslow-rl/runs/7d9ztg9q)

## Checkpoints Available

- `final_model/` - Merged LoRA weights (best for inference)
- `checkpoint-250/` - Mid-training checkpoint
- `checkpoint-500/` - Final training checkpoint
- `sample_completions.jsonl` - Sample outputs during training

## Hierarchical Reward Structure

**Tier A (Structure):**
- Parsable JSON (0.3 weight)
- Valid schema with reasoning steps (0.4 weight)
- Numeric answer present (0.2 weight)
- JSON-only output (0.1 weight)

**Tier B (Correctness):**
- Binary reward for correct answer

**Gating Function:**
```python
gate_b = sigmoid(k * (r_a - tau))
total_reward = r_a + beta * gate_b * r_b
```
where k=3, tau=0.5, beta=1.0

This ensures the model must achieve high structure quality (r_a > 0.5) before correctness rewards (r_b) significantly impact training.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load trained LoRA adapter
model = PeftModel.from_pretrained(base_model, "aaron1729/maslow-rl-gsm8k-gated")
tokenizer = AutoTokenizer.from_pretrained("aaron1729/maslow-rl-gsm8k-gated")

# Generate
messages = [
    {"role": "system", "content": "You must output a JSON object with step_1, step_2, etc. for reasoning and answer for the final numeric answer."},
    {"role": "user", "content": "Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. How many does she sell daily?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training Details

- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Training data**: 500 samples from GSM8K
- **Eval data**: 100 samples from GSM8K
- **Training steps**: 500
- **Batch size**: 8 problems × 8 generations = 64 completions/step
- **Learning rate**: 1e-4
- **LoRA config**: r=16, alpha=32, dropout=0.05
- **Temperature**: 1.0 (increased for exploration)
- **KL coefficient**: 0.01

## Repository

Full code and configs: [aaron1729/maslow-RL](https://github.com/aaron1729/maslow-RL)

## Framework Versions

- TRL: 0.26.2
- Transformers: 4.57.3
- PyTorch: 2.9.1
- PEFT: 0.14.0

## Citation

```bibtex
@misc{maslow-rl-2025,
  title={Maslow RL: Hierarchical Reward Shaping for Mathematical Reasoning},
  author={Aaron Scher},
  year={2025},
  url={https://github.com/aaron1729/maslow-RL}
}
```
