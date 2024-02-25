import json
import re
from typing import NoReturn

from tqdm import tqdm

from model import create_client
from prompts import system_prompt_decompose_task, user_prompt_decompose_task


def create_dataset_rows(count: int = 0, N: int = 25) -> NoReturn:
    """Create rows of dataset.

    Args:
        count (int, optional): Count number. Defaults to 0.
        N (int, optional): Number of dataset files each with given rows. Defaults to 25.

    Returns:
        NoReturn:
    """

    client = create_client()

    for i in tqdm(range(N)):

        file_name = f"task_decomposition_dataset_{count+i}.json"

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_decompose_task,
                },
                {"role": "user", "content": user_prompt_decompose_task},
            ],
            temperature=1,
        )

        result = response.choices[0].message.content

        result_match = re.search("```json", result)

        if result_match:
            result = json.loads(result.strip("```json"))
        else:
            result = json.loads(result)

        with open(f"data/{file_name}", "w") as file:
            json.dump(result, file, indent=4)
