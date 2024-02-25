import json
import re

from tqdm import tqdm

from dataset import custom_dataset_load, split_train_test_dataset
from prompts import add_column


def main():

    data = custom_dataset_load("./data")
    new_data = add_column(data)

    _, test_data = split_train_test_dataset(new_data)

    N_test = len(test_data)

    with open("results/eval_responses.json", "r") as json_file:
        results = json.load(json_file)

    for i in tqdm(range(N_test)):
        text = results[i]

        pattern = "### Answer:"
        pattern_sub = "##"

        # Using re.search() to find the pattern
        match = re.search(pattern, text)
        if match:
            # Get the start and end positions of the match
            _, end_pos = match.span()

            following_text = text[end_pos:]
            sub_match = re.search(pattern_sub, following_text)
            if sub_match:
                start_sub_pos, _ = sub_match.span()
                prediction_result = following_text[:start_sub_pos]
            else:
                prediction_result = following_text

            print(f"\n\nPREDICTED_RESULT_{i}:\n\n, {prediction_result}")

        else:
            print("Pattern not found.")


if __name__ == "__main__":
    main()
