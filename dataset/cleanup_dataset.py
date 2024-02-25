import json
import os


def modify_json_structure(directory, file_names):
    # Iterate through each file in the specified directory
    for filename in os.listdir(directory):
        if filename in file_names:
            file_path = os.path.join(directory, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    # Load the JSON data from the file
                    data = json.load(file)

                    # Modify each dictionary in the list as per the new structure
                    for item in data:
                        item["output"] = " ".join(item["output"])

                # Write the modified list of dictionaries back to the JSON file
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)

                print(f"Successfully modified {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# # Specify the directory containing the JSON files
# data_directory = "./data"
# file_names = [
#     # "task_decomposition_dataset_24.json",
#     # "task_decomposition_dataset_27.json",
#     "task_decomposition_dataset_38.json",
# ]

# # Call the function with the path to the directory
# modify_json_structure(data_directory, file_names)
