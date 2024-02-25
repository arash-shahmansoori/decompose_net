system_prompt_decompose_task = """
You are a helpful assistant that creates dataset for fine-tuning LLMs according to the following instructions.

### INSTRUCTIONS
Create EXACTLY 20 rows in dataset, for generating content and analyzing the result for multimodal tasks, examples according to the following instructions:

Notes:
    1) multi-modal means the composite task may involve a combination of audio, image, text, and video.
    2) make sure the generation process may be related to one or more of the above modes, e.g., image, text, audio, and video
    3) make sure the dataset includes both single mode and multi mode scenarios for the above modes. Single mode scenario is like: generate the image and
    analyze the result etc. Multimode scenraio is a combination of the above modes.
    4) composite task should be comprised of generation of subtask and analysis of subtask.
    5) there should be only two subtasks one for generation of content another for analysis.

    For each task you're given, you are to decompose it into more manageable subtasks. Provide clear instructions, input, and a detailed output that showcases the breakdown of the composite task. Use the format below to structure your responses as a list of dictionaries:
   [
   ...,
    {
        "instruction": "Describe the composite task that needs to be broken down.",
        "output": "List all the subtasks derived from the composite task, clearly labeling them by numbers only using the same format of enumeration, in a sequential manner. Only output the subtask no additional details or fields.",
    },
    ...
    ]
"""
