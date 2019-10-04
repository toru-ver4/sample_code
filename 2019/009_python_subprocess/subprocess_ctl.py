#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
subprocess を使って exe ファイルと通信する。
"""

# import libraries
import os
from subprocess import Popen, PIPE
from threading import Thread
import time

# define
STDOUT_THREAD_TIMEOUT_SEC = 1.0


def print_stdout(proc):
    while True:
        line = proc.stdout.readline()
        print(line.decode('utf-8').rstrip())
        if not line and proc.poll() is not None:
            break


class SubprocessCtrl():
    def __init__(self, cmd_args="./bin/subprocess.exe"):
        self.proc = Popen(cmd_args, stdin=PIPE, stdout=PIPE)
        self.stdout_thread = Thread(target=print_stdout, args=(self.proc,))
        self.stdout_thread.start()

    def exit(self):
        self.stdout_thread.join(timeout=STDOUT_THREAD_TIMEOUT_SEC)

    def send_str(self, strings=b"EOFEOF\n"):
        """
        子プロセスの stdin に文字列を送る。
        文字列はあらかじめ binary にしておくこと。
        """
        self.proc.stdin.write(strings)
        self.proc.stdin.flush()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_communication_exe_files()
    p = SubprocessCtrl()
    p.send_str(b"omaehadareda\n")
    time.sleep(2.0)
    p.send_str(b"chikubi\n")
    time.sleep(2.0)
    p.send_str(b"EOFEOF\n")
    time.sleep(2.0)
    p.exit()
