from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

print(model)


import bitsandbytes as bnb


def find_all_linear_names(model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
    return list(lora_module_names)


modules = find_all_linear_names(model)
print(modules)


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


trainable, total = model.get_nb_trainable_parameters()
print(
    f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%"
)


from huggingface_hub import notebook_login

notebook_login()


# from datasets import load_dataset
# data = load_dataset("ronal999/finance-alpaca-demo", split='train')
# data = data.train_test_split(test_size=0.1)
# train_data = data["train"]
# test_data = data["test"]


# import transformers

# tokenizer.pad_token = tokenizer.eos_token


# trainer = transformers.Trainer(
#     model=model,
#     train_dataset=train_data,
#     eval_dataset=test_data,
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         warmup_steps=0.03,
#         max_steps=100,
#         learning_rate=2e-4,
#         fp16=True,
#         logging_steps=1,
#         output_dir="outputs_mistral_b_finance_finetuned_test",
#         optim="paged_adamw_8bit",
#         save_strategy="epoch",
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )
