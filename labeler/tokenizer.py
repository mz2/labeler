from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore

_tokenizer = None


def get_tokenizer(tokenizer_model: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    global _tokenizer
    if _tokenizer is None or _tokenizer.name_or_path != tokenizer_model:
        _tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model, clean_text=True, strip_accents=True, unk_token="[UNK]"
        )
    return _tokenizer


def tokenized_text(
    preprocessed_text: str, chunk_size: int, tokenizer_model: str, show_boundaries: Optional[bool] = False
) -> str:
    tokenizer = get_tokenizer(tokenizer_model)
    output = ""

    for i in range(0, len(preprocessed_text), chunk_size):
        chunk = preprocessed_text[i : i + chunk_size]
        tokens = tokenizer.tokenize(chunk)  # type: ignore
        output += " ".join(tokens)
        if show_boundaries:
            output += "\n--- Chunk boundary ---\n"

    return output
