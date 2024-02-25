from typing import NoReturn

from transformers.models.mistral.modeling_mistral import MistralForCausalLM


def print_trainable_parameters(model: MistralForCausalLM) -> NoReturn:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    # Iterate through model parameters
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # Print the number of trainable parameters, total parameters, and the percentage of trainable parameters
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
