from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


def get_completion(query: str, model:MistralForCausalLM, tokenizer:LlamaTokenizerFast, device:str="cuda:0") -> str:

    prompt_template = """
    Below is an instruction paired with an input that outlines a composite task. Your response should fulfill the instruction by providing a structured breakdown of the task into distinct subtasks.
    ### Question:
    {query}

    ### Answer:
    """
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1000,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]
