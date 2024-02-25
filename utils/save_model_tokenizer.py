import gc
from typing import NoReturn

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_model_and_tokenizer(model_name: str, new_model: str) -> NoReturn:
    # Save artifacts
    trainer.model.save_pretrained("final_checkpoint")
    tokenizer.save_pretrained("final_checkpoint")

    # Flush memory
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, "final_checkpoint")
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)

    # Push them to the HF Hub
    # model.push_to_hub(new_model, use_temp_dir=False, token=hf_token)
    # tokenizer.push_to_hub(new_model, use_temp_dir=False, token=hf_token)
