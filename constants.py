from pathlib import Path


WORK_DIR = Path(__file__).parent
DATA_DIR = (WORK_DIR / "../data/").resolve()

DIGIT_WORD_MAP_PATH = DATA_DIR / "WORDMAP_digits.json"
SVHN_EVAL_PATH = WORK_DIR / "svhn_evaluation.csv"
