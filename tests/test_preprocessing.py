from pathlib import Path
from labeler.preprocessor import preprocessed_text
from labeler.tokenizer import tokenized_text


def test_preprocess_text():
    input_text = "Visit http://example.com or 192.168.1.1 for more info. This number is 3.14159."
    expected_output = "Visit [URL] or [IP_ADDRESS] for more info. This number is [FLOAT]."
    assert preprocessed_text(input_text) == expected_output


def test_tokenize_file(tmp_path: Path):
    input_text = "This is a sample text for tokenization."
    expected_output = "this is a sample text for token ##ization ."
    tokenizer_model = "bert-base-uncased"

    assert tokenized_text(str(input_text), 512, tokenizer_model) == expected_output
