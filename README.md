# labeler - ML driven labeler of files

## Prerequisites

- poetry

## Running instructions

```bash
poetry install
poetry run python labeler/cli/main.py train --labels tests/fixtures/doctype/labels.csv --logs ./tests/fixtures/doctype --save tests/fixtures/doctype.model
```
