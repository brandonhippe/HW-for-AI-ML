from collections import deque
from copy import deepcopy
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from rapidfuzz import fuzz

BASE_DIR = Path(__file__).parent.parent.parent

sys.path.insert(0, str(BASE_DIR))
from aoc import get_released


def filled_in(string: str, **kwargs) -> str:
    filled_in = string[:]
    while replace_match := re.search(r"#\{(\w+)\}", filled_in):
        var_name = replace_match.group(1)
        filled_in = filled_in.replace(replace_match.group(0), str(kwargs[var_name]))
    return filled_in


def main():
    fuzz_threshold = 90

    assert len(sys.argv) > 1, "No input path provided"
    template_path = list(filter(lambda w: sys.argv[1] in w[-1], os.walk(Path(__file__).parent)))
    assert len(template_path) == 1, f"File not found: {template_path}"
    root, _, _ = template_path[0]
    template_path = Path(root, sys.argv[1])
    
    template_diffs = subprocess.run(["diff", "-a", "-u", "-d", "-N", str(template_path), str(Path(BASE_DIR, sys.argv[1]))], capture_output=True, text=True).stdout.split("\n")
    print("\n".join(template_diffs))

    template_removals = deque()
    template_additions = deque()

    diff_regex = r"@@ -(\d+),\d+ \+(\d+),\d+ @@"
    for line in template_diffs[2:]:
        if group_start := re.search(diff_regex, line):
            old_line = int(group_start.group(1))
            new_line = int(group_start.group(2))
        else:
            if line.startswith("-"):
                template_removals.append((old_line, line))
                old_line += 1
            elif line.startswith("+"):
                template_additions.append((new_line, line))
                new_line += 1
            else:
                old_line += 1
                new_line += 1

    for year in get_released():
        for day in get_released(year):
            expected_filename = filled_in(template_path.name, **locals())

            match_files = list(filter(lambda w: expected_filename in w[-1] and re.search(rf"{day}[.\/]", str(Path(w[0], expected_filename))), os.walk(Path(BASE_DIR, f"{year}"))))
            if len(match_files) != 1:
                continue

            root, _, _ = match_files[0]
            filepath = Path(root, expected_filename)
            assert os.path.exists(filepath), f"File not found: {str(filepath)}"
            print(f"Updating {str(filepath)}")

            with open(filepath, "r") as f:
                file_lines = [line.strip('\n') for line in f.readlines()]

            def first_close_match(line: str, change: List[Any]) -> Optional[Tuple[int, str]]:
                if len(set(line)) < 2:
                    return None
                
                for i, change_line in enumerate(change):
                    if isinstance(change_line, str):
                        change_line = (i, change_line)
                    if fuzz.ratio(line, change_line[1]) > fuzz_threshold:
                        return change_line

            additions = deepcopy(template_additions)
            removals = deepcopy(template_removals)

            insert_offset = -1
            new_file_lines = []
            for line in file_lines:
                if rem_match := first_close_match(line, removals):
                    match_offset = len(new_file_lines) - rem_match[0]
                    while (removal := removals.popleft()) != rem_match:
                        if m := first_close_match(removal[1], new_file_lines):
                            new_file_lines.pop(m[0])
                            match_offset -= 1

                    if add_match := first_close_match(line, additions):
                        insert_offset = 0 if add_match[0] < rem_match[0] else match_offset
                        insert_offset -= 1
                        
                        while (addition := additions.popleft()) != add_match:
                            new_file_lines.insert(addition[0] + insert_offset, addition[1][1:])
                        new_file_lines.insert(add_match[0] + insert_offset, add_match[1][1:])
                else:
                    new_file_lines.append(line)

            while additions:
                addition = additions.popleft()
                new_file_lines.insert(addition[0] + match_offset, addition[1][1:])

            # print("-" * os.get_terminal_size().columns)
            # print("\n".join(new_file_lines))
            # print("-" * os.get_terminal_size().columns)
            # return
            with open(filepath, "w") as f:
                f.write("\n".join(new_file_lines))


if __name__ == "__main__":
    main()
