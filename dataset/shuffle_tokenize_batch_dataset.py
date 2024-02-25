from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from type_extensions import DataPointType


def shuffle_tokenize_batch(
    data: DataPointType, tokenizer: LlamaTokenizerFast
) -> DataPointType:
    data = data.shuffle(seed=1234)  # Shuffle dataset here
    data = data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
    return data
