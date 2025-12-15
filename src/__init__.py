from pathlib import Path

SEED = 47

BASE_MODEL = "meta-llama/Meta-Llama-3.1-405B"
CHAT_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct"

# Generation configs (OpenAI-compatible)
BASE_MODEL_CONFIG = {
    "temperature": 0.0,  # Deterministic for reproducibility
    "max_tokens": 5,  # Only need 1 token for label, but allow a few for safety
    "top_p": 1.0,  # No nucleus sampling (use all probability mass)
    "frequency_penalty": 0.0,  # No penalty for token frequency
    "presence_penalty": 0.0,  # No penalty for token presence
    "logprobs": 5,  # Return top 5 logprobs for probability extraction
}

CHAT_MODEL_CONFIG = {
    "temperature": 0.0,  # Deterministic for reproducibility
    "max_tokens": 10,  # Increased to avoid empty responses from API
    "top_p": 1.0,  # No nucleus sampling
    "frequency_penalty": 0.0,  # No penalty for token frequency
    "presence_penalty": 0.0,  # No penalty for token presence
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "truthfulqa_train.json"
TEST_DATA_PATH = DATA_DIR / "truthfulqa_test.json"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ZERO_SHOT_BASE_RESULT_PATH = RESULTS_DIR / "zero_shot_base.json"
ZERO_SHOT_CHAT_RESULT_PATH = RESULTS_DIR / "zero_shot_chat.json"
FEW_SHOT_GOLDEN_RESULT_PATH = RESULTS_DIR / "few_shot_golden.json"
FEW_SHOT_ICM_RESULT_PATH = RESULTS_DIR / "few_shot_icm.json"

MAIN_FIG_PATH = RESULTS_DIR / "comparison_figure.png"
