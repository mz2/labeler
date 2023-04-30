from typing import List
import pytest
from parser.processor import filter_params, filter_uninteresting_lines


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
    assert filter_uninteresting_lines(input_lines, n=1) == expected_output


def test_filter_debug_lines_with_two_windows():
    input_lines = ["line 1\n", "line 2\n", "line 3\n", "line 4 DEBUG line\n", "line 5 DEBUG line\n", "line 6\n"]
    expected_output = ["line 1\n", "line 2\n", "line 3\n", "line 4 DEBUG line\n", "line 5 DEBUG line\n", "line 6\n"]
    assert filter_uninteresting_lines(input_lines, n=1) == expected_output


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
    assert filter_uninteresting_lines(input_lines, n=1) == expected_output


def test_filter_debug_lines_with_two_windows_and_a_line_in_between_to_trim():
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
    assert filter_uninteresting_lines(input_lines, n=1) == expected_output


def test_filter_debug_lines():
    log_lines = [
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "INFO [root@<IP>]: Postgres service started",
        "DEBUG [root@<IP>]: systemctl start postgresql",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres data directory cleaned",
        "DEBUG [root@<IP>]: sudo -u postgres pg_basebackup -h -D /var/lib/postgresql/12/main -v --wal-method=stream",
        "DEBUG [root@<IP>]: sudo -u postgres pg_basebackup -h -D /var/lib/postgresql/12/main -v --wal-method=stream",
        "INFO [root@<IP>]: Postgres replication set up successfully",
        "DEBUG [root@<IP>]: echo 'manual' > /etc/postgresql/12/main/start.conf",
        "DEBUG [root@<IP>]: echo 'manual' > /etc/postgresql/12/main/start.conf",
        "DEBUG [root@<IP>]: echo 'manual' > /etc/postgresql/12/main/start.conf",
        "INFO [root@<IP>]: Postgres service disabled",
        "DEBUG [root@<IP>]: systemctl disable postgresql.service",
        "DEBUG [root@<IP>]: systemctl disable postgresql@12-main.service",
        "DEBUG [root@<IP>]: systemctl disable postgresql@12-main.service",
        "DEBUG [root@<IP>]: systemctl disable postgresql@12-main.service",
        "INFO [root@<IP>]: Pacemaker and Corosync installed",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
    ]

    expected_lines_window_1 = [
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "INFO [root@<IP>]: Postgres service started",
        "DEBUG [root@<IP>]: systemctl start postgresql",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres data directory cleaned",
        "DEBUG [root@<IP>]: sudo -u postgres pg_basebackup -h -D /var/lib/postgresql/12/main -v --wal-method=stream",
        "DEBUG [root@<IP>]: sudo -u postgres pg_basebackup -h -D /var/lib/postgresql/12/main -v --wal-method=stream",
        "INFO [root@<IP>]: Postgres replication set up successfully",
        "DEBUG [root@<IP>]: echo 'manual' > /etc/postgresql/12/main/start.conf",
        "DEBUG [root@<IP>]: echo 'manual' > /etc/postgresql/12/main/start.conf",
        "INFO [root@<IP>]: Postgres service disabled",
        "DEBUG [root@<IP>]: systemctl disable postgresql.service",
        "DEBUG [root@<IP>]: systemctl disable postgresql@12-main.service",
        "INFO [root@<IP>]: Pacemaker and Corosync installed",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
    ]

    expected_lines_window_2 = [
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "INFO [root@<IP>]: Postgres service started",
        "DEBUG [root@<IP>]: systemctl start postgresql",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres data directory cleaned",
        "DEBUG [root@<IP>]: sudo -u postgres pg_basebackup -h -D /var/lib/postgresql/12/main -v --wal-method=stream",
        "DEBUG [root@<IP>]: sudo -u postgres pg_basebackup -h -D /var/lib/postgresql/12/main -v --wal-method=stream",
        "INFO [root@<IP>]: Postgres replication set up successfully",
        "DEBUG [root@<IP>]: echo 'manual' > /etc/postgresql/12/main/start.conf",
        "DEBUG [root@<IP>]: echo 'manual' > /etc/postgresql/12/main/start.conf",
        "DEBUG [root@<IP>]: echo 'manual' > /etc/postgresql/12/main/start.conf",
        "INFO [root@<IP>]: Postgres service disabled",
        "DEBUG [root@<IP>]: systemctl disable postgresql.service",
        "DEBUG [root@<IP>]: systemctl disable postgresql@12-main.service",
        "DEBUG [root@<IP>]: systemctl disable postgresql@12-main.service",
        "DEBUG [root@<IP>]: systemctl disable postgresql@12-main.service",
        "INFO [root@<IP>]: Pacemaker and Corosync installed",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
    ]

    filtered_lines_window_1 = filter_uninteresting_lines(log_lines, 1)
    filtered_lines_window_2 = filter_uninteresting_lines(log_lines, 2)

    assert filtered_lines_window_1 == expected_lines_window_1
    assert filtered_lines_window_2 == expected_lines_window_2


def test_filter_in_between_non_debug_lines():
    log_lines = [
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres service started",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres data directory cleaned",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
    ]

    expected_lines_window_1 = [
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres service started",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres data directory cleaned",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
    ]

    expected_lines_window_3 = [
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres service started",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
        "INFO [root@<IP>]: Postgres data directory cleaned",
        "DEBUG [root@<IP>]: rm -rf /var/lib/postgresql/12/main",
    ]

    filtered_lines_window_1 = filter_uninteresting_lines(log_lines, 1)
    filtered_lines_window_3 = filter_uninteresting_lines(log_lines, 3)
    assert filtered_lines_window_1 == expected_lines_window_1
    assert filtered_lines_window_3 == expected_lines_window_3


def test_filtering_with_another_large_input():
    input_str = [
        "249 wal_keep_segments = <NUM>' >> /etc/postgresql/<NUM>/main/<URL> ['postgresql.conf']",
        "250 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "250 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'systemctl', 'start', 'postgresql']",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'rm', '-rf', '/var/lib/postgresql/12/main']",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'rm', '-rf', '/var/lib/postgresql/12/main']",
        "251 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: sudo -u postgres <*> <*> <*> <*> <*> <*> <*> ['pg_basebackup', '-h', '-D', '/var/lib/postgresql/12/main', '-v', '--wal-method=stream']",
        "251 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: sudo -u postgres <*> <*> <*> <*> <*> <*> <*> ['pg_basebackup', '-h', '-D', '/var/lib/postgresql/12/main', '-v', '--wal-method=stream']",
        "234 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: <*> <*> <*> <*> ['echo', '\"manual\"', '>', '/etc/postgresql/12/main/start.conf']",
        "234 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: <*> <*> <*> <*> ['echo', '\"manual\"', '>', '/etc/postgresql/12/main/start.conf']",
        "234 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: <*> <*> <*> <*> ['echo', '\"manual\"', '>', '/etc/postgresql/12/main/start.conf']",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'systemctl', 'disable', 'postgresql.service']",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'systemctl', 'disable', 'postgresql@12-main.service']",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'systemctl', 'disable', 'postgresql@12-main.service']",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'systemctl', 'disable', 'postgresql@12-main.service']",
        "252 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "252 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "252 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "219 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root INFO <*> -> <*> ['/tmp/tmpwj8b80wj', '<IP>:/etc/corosync/corosync.conf']",
        "219 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root INFO <*> -> <*> ['/tmp/tmpxjifdjmo', '<IP>:/etc/corosync/corosync.conf']",
        "219 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root INFO <*> -> <*> ['/tmp/tmpced_zyug', '<IP>:/etc/corosync/corosync.conf']",
    ]

    expected_str = [
        "249 wal_keep_segments = <NUM>' >> /etc/postgresql/<NUM>/main/<URL> ['postgresql.conf']",
        "250 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "250 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: install -o postgres -g postgres -m <NUM> -d /var/lib/postgresql/<NUM>/main/tmp",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'systemctl', 'start', 'postgresql']",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'rm', '-rf', '/var/lib/postgresql/12/main']",
        "26 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG <*> <*> <*> <*> ['[root@<IP>]:', 'systemctl', 'disable', 'postgresql@12-main.service']",
        "252 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "252 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "252 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root DEBUG [root@<IP>]: DEBIAN_FRONTEND=noninteractive apt-get -q install -y pacemaker corosync crmsh",
        "219 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root INFO <*> -> <*> ['/tmp/tmpwj8b80wj', '<IP>:/etc/corosync/corosync.conf']",
        "219 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root INFO <*> -> <*> ['/tmp/tmpxjifdjmo', '<IP>:/etc/corosync/corosync.conf']",
        "219 <NUM>-<NUM>-<NUM>-<NUM>:<NUM>:<NUM> root INFO <*> -> <*> ['/tmp/tmpced_zyug', '<IP>:/etc/corosync/corosync.conf']",
    ]

    assert filter_uninteresting_lines(input_str, 4) == expected_str
