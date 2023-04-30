from typing import List
import pytest
from parser.processor import filter_params, filter_debug_lines


@pytest.mark.parametrize(
    "params, expected",
    [
        (["subprocess.CalledProcessError", "3", "2", "1"], ["subprocess.CalledProcessError"]),
        (["status.bash"], ["status.bash"]),
        (["console.out"], ["console.out"]),
        (["console.out", "-1", "2", "1", "time.out"], ["console.out", "time.out"]),
        (["1", "5", "1"], []),
        (["433a9b11-fc2b-61c7-9e3b-ee6a1b6f4700"], []),  # UUID-like value
        (["status.bash", "433a9b11-fc2b-61c7-9e3b-ee6a1b6f4700"], ["status.bash"]),
        (["not_an_ip_address"], ["not_an_ip_address"]),
        (["10.245.218.44"], []),
        (["10.245.208.73"], []),
        (["root@10.246.64.5"], ["root@<IP>"]),
        (["foobar://root:lalala@123.246.64.5:8080/herp/derp"], ["foobar://root:lalala@<IP>:8080/herp/derp"]),
        (["text_with_123.246.64.5_and_more"], ["text_with_<IP>_and_more"]),
        (["[root@10.246.64.7]:"], ["[root@<IP>]:"]),
    ],
)
def test_filter_unwanted_params(params: List[str], expected: List[str]):
    filtered_params = filter_params(params)
    assert filtered_params == expected


def test_filter_debug_lines_simple():
    input_lines = ["line 1\n", "line 2\n", "line 3\n", "line 4 DEBUG line\n", "line 5 DEBUG line\n"]
    expected_output = ["line 1\n", "line 2\n", "line 3\n", "line 4 DEBUG line\n"]
    assert filter_debug_lines(input_lines, n=1) == expected_output


def test_filter_debug_lines_with_two_windows():
    input_lines = ["line 1\n", "line 2\n", "line 3\n", "line 4 DEBUG line\n", "line 5 DEBUG line\n", "line 6\n"]
    expected_output = ["line 1\n", "line 2\n", "line 3\n", "line 4 DEBUG line\n", "line 5 DEBUG line\n", "line 6\n"]
    assert filter_debug_lines(input_lines, n=1) == expected_output


def test_filter_debug_lines_with_two_windows_and_tail_to_trim():
    input_lines = [
        "line 1\n",
        "line 2\n",
        "line 3\n",
        "line 4 DEBUG line\n",
        "line 5 DEBUG line\n",
        "line 6\n",
        "line 7 DEBUG line\n",
        "line 8 DEBUG line\n",
    ]
    expected_output = [
        "line 1\n",
        "line 2\n",
        "line 3\n",
        "line 4 DEBUG line\n",
        "line 5 DEBUG line\n",
        "line 6\n",
        "line 7 DEBUG line\n",
    ]
    assert filter_debug_lines(input_lines, n=1) == expected_output


def test_filter_debug_lines_with_two_windows_and_tail_to_trim():
    input_lines = [
        "line 1\n",
        "line 2\n",
        "line 3\n",
        "line 4 DEBUG line\n",
        "line 5 DEBUG line\n",
        "line 6 DEBUG line\n",
        "line 7\n",
        "line 8 DEBUG line\n",
        "line 9 DEBUG line\n",
    ]
    expected_output = [
        "line 1\n",
        "line 2\n",
        "line 3\n",
        "line 4 DEBUG line\n",
        "line 6 DEBUG line\n",
        "line 7\n",
        "line 8 DEBUG line\n",
    ]
    assert filter_debug_lines(input_lines, n=1) == expected_output
