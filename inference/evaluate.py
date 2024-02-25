from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


def custom_inference(
    model_inputs,
    model: MistralForCausalLM,
    tokenizer: LlamaTokenizerFast,
):
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1000,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]
