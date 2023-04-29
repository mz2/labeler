from typing import List, Optional


def batch_data(text: str, size: int, overlap: int, max_bytes: Optional[int] = None) -> List[str]:
    lines = text.splitlines()
    start = 0
    end = size
    batches = []

    while start < len(lines):
        end = min(end, len(lines))
        batch = "\n".join(lines[start:end])

        if max_bytes is not None:
            while len(batch.encode("utf-8")) > max_bytes and end > start:
                end -= 1
                batch = "\n".join(lines[start:end])

            print(batch)

            if overlap == 0 and len(batch.encode("utf-8")) > max_bytes:
                truncated_batch = batch[:max_bytes]
                last_newline = truncated_batch.rfind("\n")
                if last_newline != -1:
                    batch = truncated_batch[: last_newline + 1]
                else:
                    batch = truncated_batch

        if batch:
            batches.append(batch)

        start += size - overlap
        end += size - overlap

    return batches
