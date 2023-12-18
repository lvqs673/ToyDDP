import os
import json
import pickle
import numpy as np
from typing import Sequence


def read_json(file_path: str) -> object:
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: object, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def read_pkl(file_path: str) -> object:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_pkl(obj: object, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_lines(file_path: str) -> list[str]:
    lines = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_lines(lines: Sequence, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
