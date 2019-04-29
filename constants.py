from pathlib import Path

from torchvision import transforms

WORK_DIR = Path(__file__).parent
DATA_DIR = (WORK_DIR / "../data/").resolve()

DIGIT_WORD_MAP_PATH = DATA_DIR / "WORDMAP_digits.json"
SVHN_EVAL_PATH = WORK_DIR / "svhn_evaluation.csv"

MODEL_INPUT_SIZE = (100, 256)  # (height, width) Average SVHN image ratio is ~2.46.
KEEP_RATIO = True  # Keep the original image ratio or not.

# Data input normalization
normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
