from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast


def create_model_inputs(
    query: str, tokenizer: LlamaTokenizerFast, device: str = "cuda:0"
):
    prompt_template = """
      Below is an instruction paired with an input that outlines a composite task. Your response should fulfill the instruction by providing a structured breakdown of the task into distinct subtasks.
      ### Question:
      {query}

      ### Answer:
      """

    prompt = prompt_template.format(query=query)
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodeds.to(device)
    return model_inputs
