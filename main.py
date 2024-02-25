import json

from tqdm import tqdm

from config import lora_config, training_args
from dataset import (
    custom_dataset_load,
    shuffle_tokenize_batch,
    split_train_test_dataset,
)
from inference import custom_inference
from model import load_model_tokenizer
from prompts import add_column
from training import prepare_model_for_qlora_training, qlora_training
from utils import (  # print_trainable_parameters,
    create_model_inputs,
    save_model_and_tokenizer,
)


def main():

    model_name = "mistralai/Mistral-7B-v0.1"
    new_model = "Decompose_Task_Mistral-7B"

    model, tokenizer = load_model_tokenizer()

    data = custom_dataset_load("./data")
    new_data = add_column(data)

    processed_data = shuffle_tokenize_batch(new_data, tokenizer)

    train_data, test_data = split_train_test_dataset(processed_data)

    model = prepare_model_for_qlora_training(model, lora_config)

    # Print the number of trainable parameters in the model after applying LoRA
    # print_trainable_parameters(model)

    # Start QLORA fine-tuning
    _, tokenizer, model = qlora_training(
        training_args, model, tokenizer, train_data, test_data
    )

    # Evaluate the performance on the test dataset and save the results
    N_test = len(test_data)
    responses_eval = []
    for i in tqdm(range(N_test)):
        model_inputs_eval = create_model_inputs(test_data[i]["instruction"])
        response_eval = custom_inference(model_inputs_eval, model)
        responses_eval.append(response_eval)

    with open("results/eval_responses.json", "w") as json_file:
        json.dump(responses_eval, json_file, indent=4)

    # Save model and tokenizer
    save_model_and_tokenizer(model_name, new_model)


if __name__ == "__main__":
    main()
