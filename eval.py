"""
Evaluation script for Maslow RL models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import wandb

from data import load_data_from_config
from rewards import batch_compute_rewards

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_model_and_tokenizer(model_path: str, base_model_name: str = None):
    """
    Load model and tokenizer from checkpoint.

    Args:
        model_path: Path to model checkpoint (can be LoRA adapter)
        base_model_name: Base model name (if loading LoRA adapter)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if this is a local path or HuggingFace model name
    model_path_obj = Path(model_path)
    is_local_path = model_path_obj.exists()

    if is_local_path and (model_path_obj / "adapter_config.json").exists():
        # This is a LoRA adapter
        if base_model_name is None:
            # Try to read from adapter config
            with open(model_path_obj / "adapter_config.json") as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")

        if base_model_name is None:
            raise ValueError("Base model name required for LoRA adapter")

        logger.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
    else:
        # Load as full model (from HuggingFace or local path)
        logger.info(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    # Configure tokenizer for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Use left-padding for decoder-only models

    model.eval()
    return model, tokenizer


def generate_completions(
    model,
    tokenizer,
    prompts: List,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    batch_size: int = 8
) -> List[str]:
    """
    Generate completions for a list of prompts.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer
        prompts: List of prompts (can be strings or message lists)
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        batch_size: Batch size for generation

    Returns:
        List of generated completions
    """
    completions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]

        # Convert message lists to strings using chat template
        batch_prompt_texts = []
        for prompt in batch_prompts:
            if isinstance(prompt, list):
                # Apply chat template
                prompt_text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt_text = prompt
            batch_prompt_texts.append(prompt_text)

        # Tokenize
        inputs = tokenizer(
            batch_prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode (remove prompt)
        for j, output in enumerate(outputs):
            prompt_length = inputs["input_ids"][j].shape[0]
            completion_tokens = output[prompt_length:]
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            completions.append(completion)

    return completions


def evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    config: Dict,
    num_samples: int = None
) -> Dict[str, Any]:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        config: Configuration dictionary
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Dictionary of evaluation metrics
    """
    if num_samples is not None:
        eval_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))

    # Extract prompts and targets
    prompts = eval_dataset["prompt"]
    target_ints = eval_dataset["target_int"]

    # Generate completions
    eval_config = config["eval"]
    completions = generate_completions(
        model,
        tokenizer,
        prompts,
        max_new_tokens=config["grpo"]["max_new_tokens"],
        temperature=eval_config["temperature"],
        batch_size=config["training"]["per_device_eval_batch_size"]
    )

    # Compute rewards
    run_type = config["experiment"]["run_type"]
    rewards, infos = batch_compute_rewards(
        completions, target_ints, run_type, config
    )

    # Aggregate metrics
    n = len(rewards)
    metrics = {
        "num_samples": n,
        "reward_total_mean": sum(rewards) / n,
        "r_a_mean": sum(info["r_a"] for info in infos) / n,
        "r_b_mean": sum(info["r_b"] for info in infos) / n,
        "gate_b_mean": sum(info["gate_b"] for info in infos) / n,
        "accuracy": sum(info["r_b"] for info in infos) / n,
        "json_parse_rate": sum(1 for info in infos if info["tier_a_components"]["parsable_json"] > 0) / n,
        "valid_schema_rate": sum(1 for info in infos if info["tier_a_components"]["valid_schema"] > 0) / n,
        "r_a_perfect_rate": sum(1 for info in infos if info["r_a"] >= 0.99) / n,
    }

    # Save sample outputs
    samples = []
    num_to_save = min(eval_config["num_samples_to_save"], n)
    for i in range(num_to_save):
        samples.append({
            "prompt": prompts[i],
            "completion": completions[i],
            "target_int": target_ints[i],
            "reward": rewards[i],
            "info": infos[i]
        })

    return metrics, samples


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate Maslow RL model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (for LoRA adapters)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results (default: model_path/eval_results.json)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize W&B if enabled
    if config["experiment"]["wandb_enabled"]:
        wandb.init(
            project=config["experiment"]["wandb_project"],
            name=f"{config['experiment']['name']}-eval",
            config=config,
            tags=["evaluation"]
        )
        logger.info("Initialized W&B for evaluation logging")

    # Load data
    logger.info("Loading evaluation dataset...")
    _, eval_dataset = load_data_from_config(config)
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    # Evaluate
    logger.info("Running evaluation...")
    metrics, samples = evaluate_model(
        model, tokenizer, eval_dataset, config, args.num_samples
    )

    # Log to W&B if enabled
    if config["experiment"]["wandb_enabled"] and wandb.run is not None:
        # Extract distributions from samples for histogram logging
        r_a_values = [s["info"]["r_a"] for s in samples]
        r_b_values = [s["info"]["r_b"] for s in samples]
        gate_b_values = [s["info"]["gate_b"] for s in samples]
        reward_values = [s["reward"] for s in samples]

        # Extract tier A component distributions
        parsable_json_values = [s["info"]["tier_a_components"]["parsable_json"] for s in samples]
        valid_schema_values = [s["info"]["tier_a_components"]["valid_schema"] for s in samples]
        numeric_answer_values = [s["info"]["tier_a_components"]["numeric_answer"] for s in samples]
        json_only_values = [s["info"]["tier_a_components"]["json_only"] for s in samples]

        # Log aggregate metrics AND histograms
        wandb.log({
            # Summary statistics
            "eval/accuracy": metrics["accuracy"],
            "eval/r_a_mean": metrics["r_a_mean"],
            "eval/r_b_mean": metrics["r_b_mean"],
            "eval/gate_b_mean": metrics["gate_b_mean"],
            "eval/reward_total_mean": metrics["reward_total_mean"],
            "eval/json_parse_rate": metrics["json_parse_rate"],
            "eval/valid_schema_rate": metrics["valid_schema_rate"],
            "eval/r_a_perfect_rate": metrics["r_a_perfect_rate"],

            # Histograms using wandb.plot for better visualization
            "eval/r_a_histogram": wandb.plot.histogram(
                wandb.Table(data=[[v] for v in r_a_values], columns=["r_a"]), "r_a", title="R_A Distribution"
            ),
            "eval/r_b_histogram": wandb.plot.histogram(
                wandb.Table(data=[[v] for v in r_b_values], columns=["r_b"]), "r_b", title="R_B (Correctness) Distribution"
            ),
            "eval/gate_b_histogram": wandb.plot.histogram(
                wandb.Table(data=[[v] for v in gate_b_values], columns=["gate_b"]), "gate_b", title="Gate_B Distribution"
            ),
            "eval/reward_histogram": wandb.plot.histogram(
                wandb.Table(data=[[v] for v in reward_values], columns=["reward"]), "reward", title="Total Reward Distribution"
            ),
        })

        # Create a table for sample completions
        sample_table = wandb.Table(
            columns=["question", "completion", "target", "predicted", "correct", "r_a", "r_b", "reward"]
        )

        for sample in samples[:10]:  # Log first 10 samples
            # Extract predicted answer from completion
            try:
                comp = sample["completion"].strip()
                if comp.startswith("```"):
                    comp = "\n".join(comp.split("\n")[1:-1])
                parsed = json.loads(comp)
                predicted = str(parsed.get("answer", "N/A"))
            except:
                predicted = "N/A"

            # Extract question text from prompt
            prompt = sample["prompt"]
            if isinstance(prompt, list) and len(prompt) > 1:
                question = prompt[1].get("content", str(prompt))
            else:
                question = str(prompt)

            sample_table.add_data(
                question[:100],  # Truncate question for display
                sample["completion"][:200],  # Truncate for display
                sample["target_int"],
                predicted,
                sample["info"]["r_b"] == 1.0,
                sample["info"]["r_a"],
                sample["info"]["r_b"],
                sample["reward"]
            )

        wandb.log({"eval/sample_predictions": sample_table})

        logger.info("Logged evaluation results to W&B")

    # Print metrics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")
    print("=" * 80)

    # Save results
    output_path = args.output
    if output_path is None:
        # For HuggingFace models, save to a simple filename
        # For local models, save in the model directory
        model_path_obj = Path(args.model_path)
        if model_path_obj.exists():
            output_path = model_path_obj / "eval_results.json"
        else:
            # HuggingFace model - save to current directory with sanitized name
            model_name = args.model_path.replace("/", "_")
            output_path = Path(f"{model_name}_eval_results.json")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": metrics,
        "samples": samples,
        "config": config
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_path}")

    # Finish W&B run if enabled
    if config["experiment"]["wandb_enabled"] and wandb.run is not None:
        wandb.finish()
        logger.info("Finished W&B logging")

    # Print a few sample outputs
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS (first 3)")
    print("=" * 80)
    for i, sample in enumerate(samples[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"Target: {sample['target_int']}")
        print(f"Completion: {sample['completion'][:200]}...")
        print(f"R_A: {sample['info']['r_a']:.2f}, R_B: {sample['info']['r_b']:.2f}, "
              f"Total: {sample['reward']:.2f}")
        print(f"Components: {sample['info']['tier_a_components']}")


if __name__ == "__main__":
    main()
