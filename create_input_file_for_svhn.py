from collections import defaultdict
import json
from pathlib import Path
import os
import pandas
from sklearn.model_selection import train_test_split

from image_captioning.constants import DIGIT_WORD_MAP_PATH
from image_captioning.utils import create_input_files


datadir = Path(__file__).parents[1] / "data"
input_dir = datadir / "svhn"
input_dir_test = input_dir / "test"
input_dir_train = input_dir / "train"
output_folder = datadir / 'svhn_image_captioning'
dataset_json = output_folder / "dataset_svhn.json"


def create_dataset_json_description(input_dir_train, input_dir_test, dataset_json):
    images = []

    labels = defaultdict(list)
    for input_dir in (input_dir_train, input_dir_test):
        bbox_data = pandas.read_csv(next(input_dir.glob("*.csv")), index_col=0)
        for index, row in bbox_data.iterrows():
            label = row["label"]
            label = 0 if label == 10 else label
            labels[index].append((row["xmin"], str(label)))

    train_indexes, _ = train_test_split(range(33405), test_size=0.2, random_state=1994)
    train_indexes = sorted(train_indexes)
    train_val_index = 0
    train_index = 0
    for key in labels:
        if "test" in key:
            split = "test"
        else:
            if train_val_index == train_indexes[train_index]:
                train_index += 1
                split = "train"
            else:
                split = "val"
            train_val_index += 1
        image = {
            "filename": os.path.split(key)[-1],
            "filepath": os.path.split(os.path.split(key)[0])[1],
            "sentences": [{
                "tokens": [label for _, label in sorted(labels[key], key=lambda x: x[0])]}],
            "split": split,
        }
        images.append(image)

    os.makedirs(str(output_folder), exist_ok=True)
    data = {
        "dataset": "svhn",
        "images": images,
    }
    with open(str(dataset_json), "w") as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    if not dataset_json.exists():
        create_dataset_json_description(input_dir_train, input_dir_test, dataset_json)
    # Create input files (along with word map)
    create_input_files(dataset='svhn',
                       karpathy_json_path=str(dataset_json),
                       image_folder=str(input_dir),
                       captions_per_image=1,
                       min_word_freq=5,
                       output_folder=str(output_folder),
                       max_len=10,
                       word_map=str(DIGIT_WORD_MAP_PATH),
    )
