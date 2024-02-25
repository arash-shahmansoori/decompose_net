from model import create_client

client = create_client()

query = "Linkedin post for this groundbraking approach to increase my chances to land a job and attract attentions from companies and recruiters."
# query = "readme.md file of my repository"
additional_requirement = ""  # Use for Title
# additional_requirement = "The abstract should be maximum 200 words. DO NOT INCLUDE TITLE ONLY OUTPUT THE ABSTRACT."  # Use for abstract
# additional_requirement = "The introduction should be appropriate for a journal/blog paper/post. You need to specifically mention the main contributions of the blog/research paper including: (1) First open source LLM for multi-model task decomposition with generation and analysis breakdown of the composite task. (2) Design of multi-modal dataset for composite task decomposition with alpaca dataset format. (3) Open sourced the code base, adapter checkpoints, and evaluation results. You have to specifically emphasize the importance of such a model acting as a controller agent in hirarchical swarm of agents networks as an orchestrator where it is of critical importance to breaks down the composite tasks to actionable subtasks and and the dispatch the subtasks among different agents."
# additional_requirement = """The proposed method should be appropriate for a journal/blog paper/post. You need to specifically mention the main contributions of the blog/research paper including: (1) First open source LLM for multi-model task decomposition with generation and analysis breakdown of the composite task. (2) Design of multi-modal dataset for composite task decomposition with alpaca dataset format. (3) Open sourced the code base, adapter checkpoints, and evaluation results. You have to specifically emphasize the importance of such a model acting as a controller agent in hirarchical swarm of agents networks as an orchestrator where it is of critical importance to breaks down the composite tasks to actionable subtasks and and the dispatch the subtasks among different agents. The proposed method should include a pictorial viewpoint of the proposed approach to clearly describing the proposed method. In particular, the pictorial viewpoint of the proposed method should describe clearly and easily all the steps in the proposed method using a descriptive and easy to read block diagram using Mermaid Markdown DIAGRAM. Subsequently, you describe the proposed algorithm based on the block diagram in details and easy to follow way. DO NOT INCLUDE TITLE, ABSTRACT, AND INTRODUCTION ONLY OUTPUT THE PROPOSED METHOD. Finally, revise the proposed method section and make sure that it follows the mermaid markdown workflow for the proposed method and all the above description provided regarding the proposed method. Make sure the diagram and the scientific workflow of the proposed method are coherent and match each other conceptually."""  # Use for the main method
# additional_requirement = """At the begining of the results you need to mention that all the source code including, dataset, scripts, notebooks, and evaluation results are publically availabe at: my github url. The results section should include the following subsections. First, the setup including the types of models used for fine-tuning. In particular, Mistral 7b was used using QLORA for the finetuning process. All the hyperparameters are available in the aforementioned url. Yuo have to also mention that gpt-4-turbo-preview was used in designing the multi-modal task decomposition dataset. Next, we need to show the result before the fine-tuning process and after the finetuning process and emphasize on the quality of the fine-tuned model compared to jbrish useless output from the base model for multi-modal task decomposition. Moreover, we need to sho the train and validation loss in the result and emphasize on the reducing the loss and the fact that model is learning efficiently from the data for task decomposition and the fact that both validation and training loss are going down as the training progress showing the efficiency of the fine-tuning process and the quality of the data and prompts for fine-tuning the base model."""


# system_prompt = """you are the best artist and philosopher and painter in the world. You create the best prompts based on philosophical concepts to generate images. The user provide you the concept that you are supposed to use to generate the prompt to be used for text to image generation, your task is to provide the best prompt considering all the nuances for text to image generation based on the philosophical concept. You create the prompt such that the potential generated image is unique, outstanding art form with deep philosophical concepts."""


system_prompt = """You are the best scientific artificial (general) intelligence researcher and blogger in the world. You provide the best suggestions for writing scientific blogs and papers related to artificial (general) intelligence, machine learning, and deep learning. You will be asked questions by the user for generating scientific suggestions for writing blogs and scientific papers. The user asks your suggestions for writing different parts of a scientific blog and papers, i.e., abstract, title, introduction, main methods, results, and conclusions. You only respond to what the user asks for and do not try to provide additional responses that are not related to the original user query."""

user_prompt = f"""
I have fine-tuned an open source LLM named mistral with 7b parameters based on a custom dataset that I created. The fine-tuned model is the first of its kind to decompose a composite task and break it down to actionable subtasks. This of critical importance for designing the controller agents that orchestrates tasks among different agents, e.g. hirarchichal swarm network of agents. The generated dataset for fine-tuning the LLM is original and created based on alpaca type dataset format for LLMs. For the fine-tuning process itself different advanced technologies are used including Quantized Low Rank Adapters (QLORA). All the training source code and adapters are open sourced together with the dataset and the evaluation results. The dataset for task decomposition is designed such taht it follows the alpaca format and multi-model scenarios including text, image, video, and audio and combination of multi-mode and single-mode composite tasks with wide variety of complexity in terms of generation and analysis of multi-modal composite tasks.

Our main contributions are: 1) First open source LLM for task decomposition 2) Dataset for multi-modal task decomposition 3) Design of multi-modal dataset for composite task decomposition

According to the above information, provide the best possible {query} for this blog/research paper. The {query} should be smart, unique, concise, and capture all the key novel aspects of the aforementioned process above as much as possible. It should emphasize on the novel aspects and contibutions of the above process. 

{additional_requirement}

"""


response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ],
    stream=True,
)


for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
