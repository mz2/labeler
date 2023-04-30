from typing import Optional
from transformers import AutoTokenizer  # type: ignore


def tokenized_text(
    preprocessed_text: str, chunk_size: int, tokenizer_model: str, show_boundaries: Optional[bool] = False
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, clean_text=True, strip_accents=True, unk_token="[UNK]")
    output = ""

    for i in range(0, len(preprocessed_text), chunk_size):
        chunk = preprocessed_text[i : i + chunk_size]
        tokens = tokenizer.tokenize(chunk)
        output += " ".join(tokens)
        if show_boundaries:
            output += "\n--- Chunk boundary ---\n"

    return output
