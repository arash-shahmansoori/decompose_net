from transformers import DataCollatorForLanguageModeling, Trainer
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mistral.modeling_mistral import MistralForCausalLM

from type_extensions import DataPointTest, DataPointTrain


def qlora_training(
    training_args,
    model: MistralForCausalLM,
    tokenizer: LlamaTokenizerFast,
    train_data: DataPointTrain,
    test_data: DataPointTest,
):
    # Set the pad_token of the tokenizer to be the same as the eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Create a trainer for fine-tuning a model
    trainer = Trainer(
        model=model,  # The model to be trained
        train_dataset=train_data,  # Training dataset
        eval_dataset=test_data,  # Evaluation dataset
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),  # Data collator for language modeling task
    )

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()

    return trainer, tokenizer, model
