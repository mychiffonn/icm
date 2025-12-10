"""Utility functions, such as file operations or API calls."""

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import openai
from dotenv import load_dotenv

from src import BASE_MODEL, BASE_MODEL_CONFIG, CHAT_MODEL, DATA_DIR

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """Single example from TruthfulQA data subset."""

    question: str
    choice: str
    label: int
    consistency_id: int


def read_json(file_path: str | Path) -> list[dict]:
    """Read a JSON file and return a list of dictionaries.

    Args:
        file_path (str | Path): Path to the JSON file, relative to PROJECT_ROOT

    Returns:
        list[dict]: List of dictionaries parsed from the file.
    """
    fpath = (
        Path(file_path).resolve() if isinstance(file_path, str) else file_path.resolve()
    )

    if fpath.suffix != ".json":
        raise ValueError(f"The file {fpath} is not a JSON file.")
    if not fpath.exists():
        raise FileNotFoundError(f"The file {fpath} does not exist.")
    if not fpath.is_file():
        raise ValueError(f"The path {fpath} is not a file.")

    with fpath.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_hyperbolic_client() -> openai.OpenAI:
    """Create and return Hyperbolic API client.

    Returns:
        openai.OpenAI: Configured client for Hyperbolic API.

    Raises:
        ValueError: If HYPERBOLIC_API_KEY is not set in environment.
    """
    api_key = os.getenv("HYPERBOLIC_API_KEY")
    if not api_key:
        raise ValueError(
            "HYPERBOLIC_API_KEY not found in environment. "
            "Please set it in your .env file."
        )

    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.hyperbolic.xyz/v1",
    )


def load_truthqa_data(
    split: Literal["train", "test"] = "train",
) -> list[Example]:
    """Load TruthfulQA data and convert to Example objects.

    Args:
        split: Dataset split to load ('train' or 'test').

    Returns:
        list[Example]: List of Example dataclass instances.

    Raises:
        ValueError: If split is not 'train' or 'test'.
        FileNotFoundError: If the data file doesn't exist.
    """
    if split not in ["train", "test"]:
        raise ValueError(f"Split must be 'train' or 'test', got '{split}'")

    filename = f"truthfulqa_{split}.json"
    filepath = DATA_DIR / filename

    # Load raw JSON data
    raw_data = read_json(filepath)

    # Convert to Example instances
    return [
        Example(
            question=item["question"],
            choice=item["choice"],
            label=item["label"],
            consistency_id=item["consistency_id"],
        )
        for item in raw_data
    ]


def save_labeled_data(data: list[Example], filepath: Path | str) -> None:
    """Save labeled dataset to JSON.

    Args:
        data: List of Example instances to save.
        filepath: Path where to save the JSON file.

    Raises:
        ValueError: If filepath is not a JSON file.
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if filepath.suffix != ".json":
        raise ValueError(f"Filepath must be a .json file, got {filepath}")

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    data_dicts = [asdict(example) for example in data]

    # Save to JSON
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data_dicts, f, indent=2, ensure_ascii=False)


def construct_few_shot_prompt(
    examples: list[Example],
    query_example: Example,
    include_query_label: bool = False,
) -> str:
    """Construct few-shot prompt for base model.

    Args:
        examples: Context examples from D_{(x_i, y_i)}.
        query_example: The example to predict label for.
        include_query_label: If True, include the query label (for testing).

    Returns:
        Formatted prompt string ready for API.
    """
    prompt_parts = []

    # Add context examples
    for ex in examples:
        prompt_parts.append(f"Question: {ex.question}")
        prompt_parts.append(f"Answer: {ex.choice}")
        prompt_parts.append(f"Label: {ex.label}")
        prompt_parts.append("")  # Blank line separator

    # Add query example
    prompt_parts.append(f"Question: {query_example.question}")
    prompt_parts.append(f"Answer: {query_example.choice}")

    if include_query_label:
        prompt_parts.append(f"Label: {query_example.label}")
    else:
        prompt_parts.append("Label: ")

    return "\n".join(prompt_parts)


def extract_label_probs(
    client: openai.OpenAI,
    prompt: str,
    model: str,
) -> tuple[float, float]:
    """Extract normalized P(y=0) and P(y=1) from model response.

    Args:
        client: OpenAI-compatible API client.
        prompt: Few-shot prompt constructed by construct_few_shot_prompt().
        model: Model name (should be base model for raw probabilities).

    Returns:
        Tuple of (prob_0, prob_1) normalized to sum to 1.

    Raises:
        ValueError: If logprobs cannot be extracted or don't contain "0" or "1".
    """
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            **BASE_MODEL_CONFIG,
            stream=False,
        )

        # Extract logprobs from response
        if not response.choices or not response.choices[0].logprobs:
            raise ValueError("No logprobs returned by API")

        all_top_logprobs = response.choices[0].logprobs.top_logprobs

        # Try to find tokens in multiple positions (in case of leading space/newline)
        logprob_0 = None
        logprob_1 = None

        for position_idx in range(min(2, len(all_top_logprobs))):
            top_logprobs = all_top_logprobs[position_idx]

            for token, logprob in top_logprobs.items():
                token_stripped = token.strip()
                if token_stripped == "0" and logprob_0 is None:
                    logprob_0 = logprob
                elif token_stripped == "1" and logprob_1 is None:
                    logprob_1 = logprob

            # If we found both, break early
            if logprob_0 is not None and logprob_1 is not None:
                break

        # If exact tokens not found, raise error
        if logprob_0 is None or logprob_1 is None:
            available_tokens_pos0 = list(all_top_logprobs[0].keys())
            raise ValueError(
                f"Could not find tokens '0' or '1' in logprobs. "
                f"Available tokens at position 0: {available_tokens_pos0}"
            )

        # Convert log probabilities to probabilities
        prob_0 = math.exp(logprob_0)
        prob_1 = math.exp(logprob_1)

        # Normalize to sum to 1 (only considering labels 0 and 1)
        total = prob_0 + prob_1
        normalized_prob_0 = prob_0 / total
        normalized_prob_1 = prob_1 / total

        return normalized_prob_0, normalized_prob_1

    except Exception as e:
        raise ValueError(f"Error extracting label probabilities: {e}") from e
