from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from typing import Dict

def load_model_tokenizer(model_id: str = "mistralai/Mistral-7B-v0.1", device_map:Dict[str,int]={"": 0})->Tuple[MistralForCausalLM,LlamaTokenizerFast]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
    return model, tokenizer