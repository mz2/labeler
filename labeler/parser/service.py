import os
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List
from labeler.parser.processor import train, matches, filter_uninteresting_lines
from labeler.parser.miner import create_template_miner
from labeler.tokenizer import tokenized_text
from starlette.status import HTTP_401_UNAUTHORIZED

app = FastAPI()

API_KEY = os.environ["API_KEY"]
DRAIN_STATE_PATH = os.environ["DRAIN_STATE_PATH"]


def api_key_header(authorization: str = Header()):
    api_key = authorization.replace("Bearer ", "")
    if api_key != API_KEY:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key


class LogProcessingRequest(BaseModel):
    log: str
    window_size: int
    size: int
    model: str
    show_boundaries: bool
    tokenize: bool


@app.post("/process")
async def process_logs(request: LogProcessingRequest, api_key: str = Depends(api_key_header)):
    template_miner = create_template_miner(persistence_config=DRAIN_STATE_PATH)
    lines = request.log.splitlines()

    if request.window_size >= 0:
        lines = filter_uninteresting_lines(lines, request.window_size)

    train(template_miner, lines)

    results: List[str] = []
    for match in matches(lines, template_miner):
        if not request.tokenize:
            results.append(match)
        else:
            tokenized = tokenized_text(match, request.size, request.model, request.show_boundaries)
            results.append(tokenized)

    return {"results": results}
