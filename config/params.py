from peft import LoraConfig
from transformers import TrainingArguments

# Define a configuration for the LoRA (Learnable Requantization Activation) method
lora_config = LoraConfig(
    r=8,  # Number of quantization levels
    lora_alpha=32,  # Hyperparameter for LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Modules to apply LoRA to
    lora_dropout=0.05,  # Dropout probability
    bias="none",  # Type of bias
    task_type="CAUSAL_LM",  # Task type (in this case, Causal Language Modeling)
)


# Define a configuration for the training
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Batch size per device during training
    gradient_accumulation_steps=4,  # Number of gradient accumulation steps
    gradient_checkpointing=True,
    warmup_ratio=0.03,  # Number of warm-up steps for learning rate
    max_steps=200,  # Maximum number of training steps
    learning_rate=5e-5,  # Learning rate
    lr_scheduler_type="cosine",  # cosine learning rate scheduler
    bf16=True,  # Enable mixed-precision training
    logging_steps=1,  # Logging frequency during training
    output_dir="Decompose_Task_Mistral-7B",  # Directory to save output files
    optim="paged_adamw_32bit",  # Optimizer type
    save_strategy="epoch",  # Strategy for saving checkpoints
    # push_to_hub=True                    # Push to the Hugging Face model hub
    report_to="wandb",
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=5,
)
