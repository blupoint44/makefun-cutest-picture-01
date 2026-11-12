"""Cutest-picture-01 entry (Python 3.13, standard library only).

The picture is entry.png beside this file, drawn procedurally by
generate.py; this program copies it to the output directory.
Outside the grader's container, MAKEFUN_OUTPUT_DIR stands in for /output:
    mkdir -p out && MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
# The grader sets MAKEFUN_OUTPUT_DIR; /output is its default mount.
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shutil.copyfile(
        os.path.join(HERE, "entry.png"),
        os.path.join(OUTPUT_DIR, "entry.png"),
    )


if __name__ == "__main__":
    main()
