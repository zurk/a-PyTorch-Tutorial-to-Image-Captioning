from pathlib import Path


WORK_DIR = Path(__file__).parent
DATA_DIR = (WORK_DIR / "../data/").resolve()

DIGIT_WORD_MAP_PATH = DATA_DIR / "WORDMAP_digits.json"
SVHN_EVAL_PATH = WORK_DIR / "svhn_evaluation.csv"

MODEL_INPUT_SIZE = (100, 256)  # (height, width) Average SVHN image ratio is ~2.46.
KEEP_RATIO = True  # Keep the original image ratio or not.