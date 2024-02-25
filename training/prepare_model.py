from peft import get_peft_model, prepare_model_for_kbit_training
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


def prepare_model_for_qlora_training(
    model: MistralForCausalLM, lora_config
) -> MistralForCausalLM:
    # Enable gradient checkpointing for the model
    model.gradient_checkpointing_enable()

    # Prepare the model for k-bit training using the "prepare_model_for_kbit_training" function
    model = prepare_model_for_kbit_training(model)

    # Get a model with LoRA applied to it using the defined configuration
    model = get_peft_model(model, lora_config)

    return model
