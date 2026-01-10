"""
Reward computation functions for Maslow RL experiment.

Implements:
- Tier A: Structure reward (JSON validity, schema compliance)
- Tier B: Correctness reward (numerical answer correctness)
- Gating mechanism (sigmoid-based hierarchical reward)
"""

import json
import re
import math
from typing import Dict, List, Tuple, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def strip_json_fences(text: str) -> str:
    """
    Remove markdown code fences from JSON output.

    Models sometimes wrap JSON in:
    ```json
    {...}
    ```
    or
    ```
    {...}
    ```

    Args:
        text: Raw model output

    Returns:
        Content with fences stripped
    """
    text = text.strip()

    # Check for opening fence
    if text.startswith("```json") or text.startswith("```"):
        lines = text.split("\n")

        # Remove first line (opening fence)
        if lines[0].startswith("```"):
            lines = lines[1:]

        # Remove last line if it's a closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


def parse_number(value: Any) -> Optional[float]:
    """
    Parse a value as a number.

    Args:
        value: Value to parse (int, float, or string)

    Returns:
        Parsed float or None if parsing fails
    """
    # Already a number
    if isinstance(value, (int, float)):
        return float(value)

    # Try to parse string
    if isinstance(value, str):
        value = value.strip().replace(",", "")

        # Match number pattern
        match = re.match(r"^-?\d+(\.\d+)?$", value)
        if match:
            try:
                return float(value)
            except ValueError:
                return None

    return None


def validate_step_schema(parsed_json: Dict) -> Tuple[bool, str]:
    """
    Validate that JSON has proper step_N structure.

    Requirements:
    - Has at least step_1 and answer keys
    - All non-answer keys match pattern step_N (N = positive integer)
    - Step numbers form complete sequence: 1, 2, 3, ..., N
    - Exactly one answer key
    - No extra keys

    Args:
        parsed_json: Parsed JSON object

    Returns:
        Tuple of (is_valid, error_message)
    """
    keys = set(parsed_json.keys())

    # Must have answer key
    if "answer" not in keys:
        return False, "Missing 'answer' key"

    # Count answer keys
    answer_count = sum(1 for k in keys if k == "answer")
    if answer_count != 1:
        return False, f"Expected exactly 1 'answer' key, found {answer_count}"

    # Get step keys
    step_keys = [k for k in keys if k != "answer"]

    # Must have at least 2 steps
    if len(step_keys) < 2:
        return False, "Must have at least 2 steps (step_1 and step_2)"

    if "step_1" not in step_keys:
        return False, "Missing 'step_1' key"

    # Validate all step keys match pattern
    step_numbers = []
    for key in step_keys:
        match = re.match(r"^step_(\d+)$", key)
        if not match:
            return False, f"Invalid key '{key}' - expected step_N format"

        step_num = int(match.group(1))
        if step_num < 1:
            return False, f"Invalid step number {step_num} - must be positive"

        step_numbers.append(step_num)

    # Check for sequential steps (no gaps)
    step_numbers.sort()
    expected_sequence = list(range(1, len(step_numbers) + 1))

    if step_numbers != expected_sequence:
        return False, f"Step numbers not sequential: {step_numbers}"

    return True, ""


def validate_step_quality(parsed_json: Dict) -> Tuple[bool, str]:
    """
    Validate that steps contain meaningful mathematical work.

    Requirements per step:
    - Length: 15-500 characters
    - Content: Must contain at least one number OR one math operation (+, -, *, /, =)

    Args:
        parsed_json: Parsed JSON object with step keys

    Returns:
        Tuple of (is_valid, error_message)
    """
    step_keys = [k for k in parsed_json.keys() if k.startswith("step_")]

    for key in step_keys:
        content = str(parsed_json[key])

        # Length check
        if len(content) < 15:
            return False, f"{key} too short (min 15 chars)"
        if len(content) > 500:
            return False, f"{key} too long (max 500 chars)"

        # Must have numbers or math operations
        has_number = bool(re.search(r'\d', content))
        has_operation = bool(re.search(r'[+\-*/=]', content))

        if not (has_number or has_operation):
            return False, f"{key} lacks mathematical content"

    return True, ""


def compute_tier_a(completion: str, weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Compute Tier A (structure) reward.

    Components:
    1. Parsable JSON (+weight)
    2. Valid schema (+weight)
    3. Numeric answer parseable (+weight)
    4. JSON-only (no extra text) (+weight)

    Args:
        completion: Model completion string
        weights: Dictionary of component weights

    Returns:
        Tuple of (total_reward, component_scores)
    """
    components = {
        "parsable_json": 0.0,
        "valid_schema": 0.0,
        "numeric_answer": 0.0,
        "json_only": 0.0
    }

    # Strip fences
    cleaned = strip_json_fences(completion)

    # 1. Try to parse JSON
    try:
        parsed = json.loads(cleaned)
        components["parsable_json"] = weights["parsable_json"]
    except json.JSONDecodeError:
        # Failed to parse - all subsequent checks fail
        total = sum(components.values())
        return total, components

    # Check if parsed result is a dict (not a primitive)
    if not isinstance(parsed, dict):
        # Valid JSON but not an object - skip remaining checks
        total = sum(components.values())
        return total, components

    # 2. Validate schema (structure and count)
    schema_valid, error_msg = validate_step_schema(parsed)
    if schema_valid:
        # 2b. Validate step quality (meaningful content)
        quality_valid, quality_error = validate_step_quality(parsed)
        if quality_valid:
            components["valid_schema"] = weights["valid_schema"]
        # If schema valid but quality fails, don't award valid_schema points

    # 3. Check if answer is parseable as number
    if "answer" in parsed:
        answer_value = parsed["answer"]
        parsed_num = parse_number(answer_value)
        if parsed_num is not None:
            components["numeric_answer"] = weights["numeric_answer"]

    # 4. Check for JSON-only (no extra text)
    # Re-serialize to canonical form
    try:
        canonical = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)

        # Compare whitespace-stripped versions
        cleaned_stripped = "".join(cleaned.split())
        canonical_stripped = "".join(canonical.split())

        if cleaned_stripped == canonical_stripped:
            components["json_only"] = weights["json_only"]
    except Exception:
        pass

    total = sum(components.values())
    return total, components


def compute_tier_b(completion: str, target_int: int) -> float:
    """
    Compute Tier B (correctness) reward.

    Returns 1.0 if answer is correct, 0.0 otherwise.

    Args:
        completion: Model completion string
        target_int: Target integer answer

    Returns:
        Reward (0.0 or 1.0)
    """
    # Try to parse and extract answer
    cleaned = strip_json_fences(completion)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return 0.0

    # Get answer field
    if "answer" not in parsed:
        return 0.0

    answer_value = parsed["answer"]
    parsed_num = parse_number(answer_value)

    if parsed_num is None:
        return 0.0

    # Check correctness with tolerance
    if abs(parsed_num - target_int) < 1e-3:
        return 1.0

    return 0.0


def sigmoid(x: float) -> float:
    """Compute sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def compute_gate_b(r_a: float, k: float, tau: float) -> float:
    """
    Compute gating value for Tier B.

    gate_b = sigmoid(k * (R_A - tau))

    Args:
        r_a: Tier A reward
        k: Steepness parameter
        tau: Threshold parameter

    Returns:
        Gate value in [0, 1]
    """
    return sigmoid(k * (r_a - tau))


def compute_reward_linear(
    completion: str,
    target_int: int,
    weights: Dict[str, float],
    beta: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute linear baseline reward: R = R_A + beta * R_B

    Args:
        completion: Model completion
        target_int: Target answer
        weights: Tier A component weights
        beta: Weight for Tier B

    Returns:
        Tuple of (total_reward, info_dict)
    """
    r_a, tier_a_components = compute_tier_a(completion, weights)
    r_b = compute_tier_b(completion, target_int)

    total_reward = r_a + beta * r_b

    info = {
        "r_a": r_a,
        "r_b": r_b,
        "gate_b": 1.0,  # No gating in linear
        "total": total_reward,
        "tier_a_components": tier_a_components
    }

    return total_reward, info


def compute_reward_gated(
    completion: str,
    target_int: int,
    weights: Dict[str, float],
    k: float = 20.0,
    tau: float = 0.85,
    beta: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute gated (Maslow) reward: R = R_A + beta * gate_b * R_B

    Args:
        completion: Model completion
        target_int: Target answer
        weights: Tier A component weights
        k: Gate steepness
        tau: Gate threshold
        beta: Weight for Tier B

    Returns:
        Tuple of (total_reward, info_dict)
    """
    r_a, tier_a_components = compute_tier_a(completion, weights)
    r_b = compute_tier_b(completion, target_int)
    gate_b = compute_gate_b(r_a, k, tau)

    total_reward = r_a + beta * gate_b * r_b

    info = {
        "r_a": r_a,
        "r_b": r_b,
        "gate_b": gate_b,
        "total": total_reward,
        "tier_a_components": tier_a_components
    }

    return total_reward, info


def batch_compute_rewards(
    completions: List[str],
    target_ints: List[int],
    run_type: str,
    config: Dict
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Compute rewards for a batch of completions.

    Args:
        completions: List of model completions
        target_ints: List of target integers
        run_type: "linear" or "gated"
        config: Configuration dictionary

    Returns:
        Tuple of (rewards, info_dicts)
    """
    weights = config["rewards"]["tier_a_weights"]
    gating = config["rewards"]["gating"]

    rewards = []
    infos = []

    for completion, target_int in zip(completions, target_ints):
        if run_type == "linear":
            reward, info = compute_reward_linear(
                completion, target_int, weights, beta=gating["beta"]
            )
        elif run_type == "gated":
            reward, info = compute_reward_gated(
                completion, target_int, weights,
                k=gating["k"], tau=gating["tau"], beta=gating["beta"]
            )
        else:
            raise ValueError(f"Unknown run_type: {run_type}")

        rewards.append(reward)
        infos.append(info)

    return rewards, infos


if __name__ == "__main__":
    # Test reward computation
    test_cases = [
        # Valid, correct
        ('{"step_1": "First I do X", "step_2": "Then Y", "answer": 42}', 42),
        # Valid, incorrect
        ('{"step_1": "Wrong reasoning", "answer": 99}', 42),
        # Invalid JSON
        ('This is not JSON', 42),
        # Missing answer
        ('{"step_1": "Some reasoning"}', 42),
        # Wrong schema (has reason instead of steps)
        ('{"reason": "my reasoning", "answer": 42}', 42),
        # Gap in steps
        ('{"step_1": "First", "step_3": "Third", "answer": 42}', 42),
    ]

    weights = {
        "parsable_json": 0.3,
        "valid_schema": 0.4,
        "numeric_answer": 0.2,
        "json_only": 0.1
    }

    print("=== Testing Reward Functions ===\n")

    for completion, target in test_cases:
        print(f"Completion: {completion[:60]}...")
        print(f"Target: {target}")

        r_a, components = compute_tier_a(completion, weights)
        r_b = compute_tier_b(completion, target)

        print(f"R_A: {r_a:.2f} - {components}")
        print(f"R_B: {r_b:.2f}")

        gate_b = compute_gate_b(r_a, k=20, tau=0.85)
        print(f"Gate_B: {gate_b:.3f}")

        linear_reward = r_a + r_b
        gated_reward = r_a + gate_b * r_b

        print(f"Linear reward: {linear_reward:.2f}")
        print(f"Gated reward: {gated_reward:.2f}")
        print("-" * 80)
