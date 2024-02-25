from typing import Dict

from type_extensions import DataPointType


def generate_prompt(data_point: Dict[str, str]) -> str:
    """Generate input text based on a prompt, task instruction, (context info.), and answer.

    Args:
        data_point (Dict[str, str]): Data point

    Returns:
        str: prompt
    """
    text = (
        "Below is an instruction that describes a composite task."
        " Your response should fulfill the instruction by providing a structured breakdown of the composite task into distinct subtasks.\n\n"
    )
    text += f'### Instruction:\n{data_point["instruction"]}\n\n'
    text += f'### Response:\n{data_point["output"]}'
    return text


def add_column(data:DataPointType, column_name: str = "prompt")->DataPointType:
    """Add the "prompt" column in the dataset"""
    text_column = [generate_prompt(data_point) for data_point in data]
    data = data.add_column(column_name, text_column)
    return data