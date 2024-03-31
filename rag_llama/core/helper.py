# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Code for handling files"""

import os
from typing import List


def find_certain_files_under_dir(root_dir: str, file_type: str = '.txt') -> List[str]:
    """Given a root folder, find all files in this folder and it's sub folders that matching the given file type."""
    assert file_type in ['.txt', '.json', '.jsonl', '.parquet', '.zst', '.json.gz', '.jsonl.gz', '.pdf']

    files = []
    if os.path.exists(root_dir):
        for root, dirnames, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(file_type):
                    files.append(os.path.join(root, f))
    return files
