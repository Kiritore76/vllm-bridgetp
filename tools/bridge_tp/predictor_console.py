"""Display progress/errors while tee preserves the complete engine log."""

import re
import sys

EXCEPTION = re.compile(r"\b(?:\w*(?:Error|Exception)|KeyboardInterrupt|SystemExit):")
STAGES = {
    "live_auxiliary_probe": "阶段1/3：在线特征对照",
    "paired_offline_hook": "阶段2/3：离线特征对照",
    "long_coverage_12requests": "阶段3/3：12条长请求采集",
}


def display(lines, output=sys.stdout):
    traceback = False
    for raw in lines:
        line = raw.replace("\r", "").rstrip()
        for marker in ("[进度]", "[异常]"):
            if marker in line:
                print(line[line.index(marker) :], file=output, flush=True)
                break
        else:
            if line.startswith("stage="):
                stage = line.split()[0][6:]
                print("[进度] " + STAGES.get(stage, stage), file=output, flush=True)
            elif line.startswith("pilot_status="):
                print("[进度] 特征与长度诊断完成", file=output, flush=True)
            elif "Traceback (most recent call last)" in line:
                traceback = True
                print(line, file=output, flush=True)
            elif (
                traceback
                or EXCEPTION.search(line)
                or " ERROR " in line
                or " CRITICAL " in line
            ):
                print(line, file=output, flush=True)
                if EXCEPTION.search(line):
                    traceback = False


if __name__ == "__main__":
    display(sys.stdin)
