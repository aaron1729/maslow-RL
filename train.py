"""
Training script for Maslow RL experiment using GRPO.
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import pytz

from dotenv import load_dotenv
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer

from data import load_data_from_config
from rewards import batch_compute_rewards
import wandb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KLDivergenceCallback(TrainerCallback):
    """Callback to log KL divergence to W&B."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging metrics."""
        if logs is not None and wandb.run is not None:
            # Extract KL-related metrics if they exist in logs
            kl_metrics = {}
            for key, value in logs.items():
                if 'kl' in key.lower() or 'divergence' in key.lower():
                    kl_metrics[f"kl/{key}"] = value

            # Log to W&B if we found any KL metrics
            if kl_metrics:
                wandb.log(kl_metrics)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def setup_model_and_tokenizer(config: Dict):
    """
    Load model and tokenizer with LoRA.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model, tokenizer, peft_config)
    """
    model_config = config["model"]
    lora_config = config["lora"]

    logger.info(f"Loading model: {model_config['name']}")

    # Determine dtype
    if model_config["dtype"] == "bf16":
        torch_dtype = torch.bfloat16
    elif model_config["dtype"] == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        trust_remote_code=True
    )

    # Configure tokenizer for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Use left-padding for decoder-only models

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=torch_dtype,
        device_map=model_config["device_map"],
        trust_remote_code=True
    )

    # Setup LoRA
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer, peft_config


def create_reward_function(config: Dict):
    """
    Create reward function for GRPO trainer.

    Args:
        config: Configuration dictionary

    Returns:
        Reward function compatible with GRPOTrainer
    """
    run_type = config["experiment"]["run_type"]

    # Accumulate metrics across multiple reward_fn calls within a step
    # Get batch size from config to know how many calls per step
    batch_size = config["training"]["per_device_train_batch_size"]

    reward_fn_state = {
        'step_metrics': [],
        'last_logged_step': -1,
        'batch_size': batch_size
    }

    def reward_fn(completions, target_int=None, **kwargs):
        """
        Reward function called by GRPOTrainer.

        Args:
            completions: List of generated completion strings
            target_int: List of target integers from dataset
            **kwargs: Other fields from dataset (unused)

        Returns:
            List of reward values
        """

        # Extract text from message format if needed
        # GRPOTrainer may pass completions as lists of message dicts
        completion_texts = []
        for comp in completions:
            if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
                # Extract content from assistant message
                completion_texts.append(comp[0].get('content', ''))
            elif isinstance(comp, str):
                completion_texts.append(comp)
            else:
                completion_texts.append(str(comp))
                logger.warning(f"Unexpected completion format: {type(comp)}")

        # Ensure target_int is a list with same length as completions
        if target_int is None:
            target_ints = [0] * len(completions)
        elif isinstance(target_int, list):
            target_ints = target_int
        else:
            # Single value, replicate for all completions
            target_ints = [target_int] * len(completions)

        rewards, infos = batch_compute_rewards(
            completion_texts, target_ints, run_type, config
        )

        # Accumulate metrics for this call
        call_metrics = {
            'r_a': [info["r_a"] for info in infos],
            'r_b': [info["r_b"] for info in infos],
            'gate_b': [info["gate_b"] for info in infos],
            'rewards': rewards
        }

        # Track call count to detect training steps
        if not hasattr(reward_fn, 'call_count'):
            reward_fn.call_count = 0
        reward_fn.call_count += 1

        # Determine current training step (batch_size reward_fn calls per step)
        batch_size = reward_fn_state['batch_size']
        current_step = reward_fn.call_count // batch_size

        # Accumulate metrics for current step
        reward_fn_state['step_metrics'].append(call_metrics)

        # Log to W&B when we complete a step (every batch_size calls)
        if reward_fn.call_count % batch_size == 0 and current_step != reward_fn_state['last_logged_step']:
            if config["experiment"]["wandb_enabled"]:
                try:
                    import wandb
                    if wandb.run is not None:
                        # Aggregate across all calls in this step
                        all_r_a = []
                        all_r_b = []
                        all_gate_b = []
                        all_rewards = []

                        for m in reward_fn_state['step_metrics']:
                            all_r_a.extend(m['r_a'])
                            all_r_b.extend(m['r_b'])
                            all_gate_b.extend(m['gate_b'])
                            all_rewards.extend(m['rewards'])

                        wandb.log({
                            "reward/r_a_mean": sum(all_r_a) / len(all_r_a),
                            "reward/r_b_mean": sum(all_r_b) / len(all_r_b),
                            "reward/gate_b_mean": sum(all_gate_b) / len(all_gate_b),
                            "reward/accuracy": sum(all_r_b) / len(all_r_b),
                            "reward/total_mean": sum(all_rewards) / len(all_rewards),
                        })
                except Exception as e:
                    logger.warning(f"Failed to log to wandb: {e}")

            # Clear metrics for next step
            reward_fn_state['step_metrics'] = []
            reward_fn_state['last_logged_step'] = current_step

        # Log sample completions to local file every 10 calls
        if reward_fn.call_count % 10 == 0:
            # Use output_dir from outer scope (already has run ID and timestamp)
            completions_file = output_dir / "sample_completions.jsonl"
            try:
                import json
                with open(completions_file, 'a') as f:
                    # Log first 2 completions from this batch
                    for i in range(min(2, len(completion_texts))):
                        log_entry = {
                            "step": reward_fn.call_count,
                            "completion": completion_texts[i],
                            "target": target_ints[i],
                            "r_a": infos[i]["r_a"],
                            "r_b": infos[i]["r_b"],
                            "gate_b": infos[i]["gate_b"],
                            "reward": rewards[i]
                        }
                        f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to log completions to file: {e}")

        # Store per-call metrics
        reward_fn.last_metrics = {
            "reward_total_mean": sum(rewards) / len(rewards),
            "r_a_mean": sum(call_metrics['r_a']) / len(call_metrics['r_a']),
            "r_b_mean": sum(call_metrics['r_b']) / len(call_metrics['r_b']),
            "gate_b_mean": sum(call_metrics['gate_b']) / len(call_metrics['gate_b']),
            "accuracy": sum(call_metrics['r_b']) / len(call_metrics['r_b'])
        }

        return rewards

    reward_fn.last_metrics = {}
    return reward_fn


def setup_training(config: Dict, model, tokenizer, train_dataset, eval_dataset):
    """
    Setup GRPO trainer.

    Args:
        config: Configuration dictionary
        model: Model with LoRA
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Eval dataset

    Returns:
        GRPOTrainer instance
    """
    training_config = config["training"]
    grpo_config = config["grpo"]
    experiment_config = config["experiment"]

    # Setup output directory (append run ID and timestamp for uniqueness in sweeps)
    base_name = experiment_config["name"]
    if wandb.run is not None:
        timestamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
        base_name = f"{base_name}-{wandb.run.id}-{timestamp}"
    output_dir = Path(experiment_config["output_dir"]) / base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create GRPO config
    grpo_training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,  # We'll use max_steps instead
        max_steps=training_config["num_train_steps"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        # GRPO-specific (corrected parameter names)
        num_generations=grpo_config["num_generations_per_prompt"],
        max_completion_length=grpo_config["max_new_tokens"],
        temperature=grpo_config["temperature"],
        top_p=grpo_config["top_p"],
        # Logging
        report_to="wandb" if experiment_config["wandb_enabled"] else "none",
        run_name=experiment_config["name"],
        # Other
        remove_unused_columns=False,
        seed=experiment_config["seed"]
    )

    # Create reward function
    reward_fn = create_reward_function(config)

    # Create callbacks
    callbacks = []
    if experiment_config["wandb_enabled"]:
        callbacks.append(KLDivergenceCallback())

    # Create trainer - GRPO needs reward_funcs (plural)
    trainer = GRPOTrainer(
        model=model,
        args=grpo_training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],  # Must be a list
        callbacks=callbacks
    )

    return trainer, reward_fn


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train Maslow RL model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--run-type",
        type=str,
        choices=["linear", "gated"],
        help="Override run type from config"
    )
    parser.add_argument(
        "--k",
        type=float,
        help="Override gating steepness parameter (k)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        help="Override gating threshold parameter (tau)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Override correctness weight parameter (beta)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override run type if specified
    if args.run_type:
        config["experiment"]["run_type"] = args.run_type
        logger.info(f"Overriding run_type to: {args.run_type}")

    # Override sweep parameters if specified
    if args.k is not None:
        config["rewards"]["gating"]["k"] = args.k
        logger.info(f"Overriding k to: {args.k}")
    if args.tau is not None:
        config["rewards"]["gating"]["tau"] = args.tau
        logger.info(f"Overriding tau to: {args.tau}")
    if args.beta is not None:
        config["rewards"]["gating"]["beta"] = args.beta
        logger.info(f"Overriding beta to: {args.beta}")

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Initialize wandb if enabled
    if config["experiment"]["wandb_enabled"]:
        # Get wandb settings from env vars or config
        wandb_project = os.getenv("WANDB_PROJECT", config["experiment"]["wandb_project"])
        wandb_entity = os.getenv("WANDB_ENTITY")

        # Add Pacific timestamp to run name
        pacific_tz = pytz.timezone('America/Los_Angeles')
        timestamp = datetime.now(pacific_tz).strftime("%Y-%m-%d, %H:%M")
        run_name = f"{config['experiment']['name']} {timestamp}"

        wandb_kwargs = {
            "project": wandb_project,
            "name": run_name,
            "config": config
        }
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity

        wandb.init(**wandb_kwargs)
        logger.info(f"W&B: Logging to project '{wandb_project}'" +
                   (f" (entity: {wandb_entity})" if wandb_entity else ""))

    # Load data
    logger.info("Loading datasets...")
    train_dataset, eval_dataset = load_data_from_config(config)
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Setup model
    logger.info("Setting up model and tokenizer...")
    model, tokenizer, peft_config = setup_model_and_tokenizer(config)

    # Setup training
    logger.info("Setting up GRPO trainer...")
    trainer, reward_fn = setup_training(
        config, model, tokenizer, train_dataset, eval_dataset
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model (use same directory logic as create_trainer)
    base_name = config["experiment"]["name"]
    if wandb.run is not None:
        timestamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
        base_name = f"{base_name}-{wandb.run.id}-{timestamp}"
    output_dir = Path(config["experiment"]["output_dir"]) / base_name
    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    # Save config
    config_save_path = output_dir / "config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_save_path}")

    if config["experiment"]["wandb_enabled"]:
        wandb.finish()


if __name__ == "__main__":
    main()
