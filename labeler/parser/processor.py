import time
import json
import re
from typing import Dict, List, Any, Iterator
from drain3 import TemplateMiner  # type: ignore
from labeler.parser.logger import logger


def train(template_miner: TemplateMiner, lines: List[str]) -> None:  # type: ignore
    line_count = 0
    start_time = time.time()
    batch_start_time = start_time
    batch_size = 10000

    for line in lines:
        line = line.rstrip()
        result: Dict[str, Any] = template_miner.add_log_message(line)  # type: ignore
        line_count += 1
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            logger.debug(
                f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                f"{len(template_miner.drain.clusters)} clusters so far."
            )
            batch_start_time = time.time()
        if result["change_type"] != "none":
            result_json = json.dumps(result)
            logger.debug(f"Input ({line_count}): " + line)
            logger.debug("Result: " + result_json)

    time_took = time.time() - start_time
    rate = line_count / time_took
    logger.debug(
        f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
        f"{len(template_miner.drain.clusters)} clusters"
    )

    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)  # type: ignore
    for cluster in sorted_clusters:
        logger.debug(cluster)


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_uuid(value: str) -> bool:
    uuid_pattern = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    return bool(uuid_pattern.match(value))


def filter_params(params: List[str]) -> List[str]:
    filtered_params = []

    for param in params:
        # a note re: IP addresses:
        # - IP address-only params are not interesting, filtered out.
        # - IP addresses as substrings (for example in URLs), they should be masked.
        if not (param.isdigit() or is_float(param) or is_uuid(param) or is_ip_address(param)):
            param = mask_ip_address_substrings(param)
            param = mask_uuid_substrings(param)
            filtered_params.append(param)

    return filtered_params


def is_ip_address(param: str) -> bool:
    ip_pattern = r"\b(?:\d{1,3}\.){1,3}\d{1,3}\b"
    return bool(re.fullmatch(ip_pattern, param))


def mask_ip_address_substrings(param: str) -> str:
    ip_pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
    if re.search(ip_pattern, param):
        param = re.sub(ip_pattern, "<IP>", param)
    return param


def mask_uuid_substrings(param: str) -> str:
    # mask out the UUID-like string from a string like lab0-silo2-cpe-5201daa7-8340-433a-9b11-fc2b61c79e3b
    uuid_pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    if re.search(uuid_pattern, param):
        param = re.sub(uuid_pattern, "<UUID>", param)
    return param


def filter_uninteresting_lines(lines: List[str], n: int) -> List[str]:
    is_filtered_line = [True if re.search(r"\bDEBUG\b", line) else False for line in lines]
    is_filtered_line_orig = is_filtered_line.copy()

    non_filtered_indexes = [i for i, _ in enumerate(lines) if not is_filtered_line[i]]
    for i in non_filtered_indexes:
        start_index = max(0, i - n)
        end_index = min(len(lines) - 1, i + n)

        for j in range(start_index, end_index + 1):
            if is_filtered_line_orig[j]:
                is_filtered_line[j] = False

    filtered_lines = [lines[i] for i in range(len(lines)) if not is_filtered_line[i]]
    return filtered_lines


def matches(lines: List[str], template_miner: TemplateMiner) -> Iterator[str]:  # type: ignore
    for line in lines:
        line = line.rstrip()
        cluster = template_miner.match(line, full_search_strategy="always")  # type: ignore

        if cluster:
            logger.debug(f"Input: {line}")
            cluster_id_str = str(cluster.cluster_id)
            template = cluster.get_template()  # type: ignore
            params = filter_params(template_miner.get_parameter_list(template, line))

            if params:
                yield f"{cluster_id_str} {template} {params}"
            else:
                yield f"{cluster_id_str} {template}"

        else:
            logger.debug(f"Input: {line}")
            yield line
