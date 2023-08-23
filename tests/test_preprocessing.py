from pathlib import Path
from labeler.tokenizer import tokenized_text


def test_tokenize_file(tmp_path: Path):
    input_text = "This is a sample text for tokenization."
    expected_output = "this is a sample text for token ##ization ."
    tokenizer_model = "bert-base-uncased"

    assert tokenized_text(str(input_text), 512, tokenizer_model) == expected_output
