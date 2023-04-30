FROM ubuntu:22.04

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install poetry
RUN poetry install --no-dev

EXPOSE 8080

CMD ["poetry", "run", "uvicorn", "parser.service:app", "--host", "0.0.0.0", "--port", "8080"]
