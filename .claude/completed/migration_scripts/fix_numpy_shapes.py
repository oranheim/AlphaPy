#!/usr/bin/env python3
"""
Fix numpy random Generator API shape issues in test files.

The new Generator API requires tuple arguments for multidimensional arrays:
- Old: rng.standard_normal(100, 5)
- New: rng.standard_normal((100, 5))
"""

import re
from pathlib import Path


def fix_numpy_shapes_in_file(filepath):
    """Fix numpy random shape arguments in a file."""
    with open(filepath) as f:
        content = f.read()

    original_content = content

    # Fix patterns like rng.standard_normal(100, 5) -> rng.standard_normal((100, 5))
    patterns = [
        (r"rng\.standard_normal\((\d+), (\d+)\)", r"rng.standard_normal((\1, \2))"),
        (r"rng\.normal\(([^,]+), ([^,]+), (\d+)\)", r"rng.normal(\1, \2, \3)"),  # This is OK as is
        (r"rng\.uniform\(([^,]+), ([^,]+), (\d+)\)", r"rng.uniform(\1, \2, \3)"),  # This is OK as is
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # Write back if changed
    if content != original_content:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False


def main():
    """Fix all test files."""
    test_dir = Path("tests")
    test_files = list(test_dir.glob("*.py"))

    fixed_count = 0
    for test_file in test_files:
        if fix_numpy_shapes_in_file(test_file):
            print(f"Fixed {test_file}")
            fixed_count += 1

    print(f"Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
