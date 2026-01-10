"""
GSM8K dataset loading, preprocessing, and target extraction.
"""

import re
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_target_int(answer_str: str) -> Optional[int]:
    """
    Extract integer target from GSM8K answer string.

    GSM8K answers are strings like "Janet sells 16 - 3 - 4 = 9 duck eggs a day. #### 9"

    Args:
        answer_str: The answer field from GSM8K

    Returns:
        Extracted integer or None if extraction fails
    """
    # Look for #### separator
    if "####" in answer_str:
        answer_part = answer_str.split("####")[1].strip()
    else:
        answer_part = answer_str.strip()

    # Remove commas (e.g., "1,000" -> "1000")
    answer_part = answer_part.replace(",", "")

    # Extract first integer token matching -?\d+
    match = re.search(r"-?\d+", answer_part)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None

    return None


def preprocess_gsm8k(
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    train_size: int = 500,
    eval_size: int = 100,
    seed: int = 42,
    system_message: str = "",
    user_template: str = "Problem: {question}"
) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess GSM8K dataset.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        train_size: Number of training examples to sample
        eval_size: Number of eval examples to sample
        seed: Random seed for sampling
        system_message: System message for prompt formatting
        user_template: Template for user message

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading dataset {dataset_name}...")

    # Load the full training split
    full_train = load_dataset(dataset_name, dataset_config, split="train")

    # Shuffle and split
    full_train = full_train.shuffle(seed=seed)

    # Extract targets and create prompts
    processed_data = []
    dropped_count = 0

    for idx, example in enumerate(full_train):
        question = example["question"]
        answer_str = example["answer"]

        # Extract target integer
        target_int = extract_target_int(answer_str)

        if target_int is None:
            dropped_count += 1
            logger.warning(f"Dropped example {idx}: could not extract target from '{answer_str}'")
            continue

        # Format prompt as message list for GRPO chat template
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_template.format(question=question)})

        processed_data.append({
            "prompt": messages,
            "question": question,
            "target_int": target_int,
            "original_answer": answer_str
        })

    logger.info(f"Processed {len(processed_data)} examples, dropped {dropped_count}")

    # Create datasets
    if len(processed_data) < train_size + eval_size:
        logger.warning(
            f"Not enough data: requested {train_size + eval_size}, "
            f"have {len(processed_data)}"
        )

    train_data = processed_data[:train_size]
    eval_data = processed_data[train_size:train_size + eval_size]

    logger.info(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")

    # Convert to HuggingFace Dataset
    from datasets import Dataset as HFDataset
    train_dataset = HFDataset.from_list(train_data)
    eval_dataset = HFDataset.from_list(eval_data)

    return train_dataset, eval_dataset


def load_data_from_config(config: Dict) -> Tuple[Dataset, Dataset]:
    """
    Load data using configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    data_config = config["data"]
    prompt_config = config["prompt"]

    return preprocess_gsm8k(
        dataset_name=data_config["dataset_name"],
        dataset_config=data_config["dataset_config"],
        train_size=data_config["train_size"],
        eval_size=data_config["eval_size"],
        seed=data_config["sample_seed"],
        system_message=prompt_config["system_message"],
        user_template=prompt_config["user_template"]
    )


if __name__ == "__main__":
    # Test data loading
    import json

    with open("config.json") as f:
        config = json.load(f)

    train_ds, eval_ds = load_data_from_config(config)

    print("\n=== Sample Training Example ===")
    print(train_ds[0])

    print("\n=== Sample Eval Example ===")
    print(eval_ds[0])
