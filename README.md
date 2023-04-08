# labeler - ML driven labeler of files

## Prerequisites

- [poetry](https://python-poetry.org/docs/) (e.g. `apt install python3-poetry` or `brew install poetry`)

## Running instructions

To run labeler, first install the prerequisites, then run the tool with `poetry run python labeler/cli/main.py`.

A trivial example case problem is provided in the repo under `tests/fixtures/doctype` with a bunch of HTML, Markdown JSON, YAML files labeled in `tests/fixtures/doctype/labels.csv` with their respective file types.

```bash
poetry install
poetry run python labeler/cli/main.py train --labels tests/fixtures/doctype/labels.csv --logs ./tests/fixtures/doctype --save tests/fixtures/doctype.model
```
