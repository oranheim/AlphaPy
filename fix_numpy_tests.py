#!/usr/bin/env python3
"""
Comprehensive script to fix all numpy.random NPY002 violations in test files.

This script systematically replaces legacy numpy.random calls with modern Generator API,
ensuring reproducibility and type safety across all test files.
"""

import re
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


def get_npy002_violations():
    """Get all NPY002 violations in test files."""
    stdout, stderr, code = run_command("uv run ruff check tests/ --select NPY002 --output-format=json")
    if code != 0:
        print(f"Error getting violations: {stderr}")
        return []

    import json

    try:
        violations = json.loads(stdout)
        return violations
    except json.JSONDecodeError:
        print(f"Could not parse ruff output: {stdout}")
        return []


def fix_numpy_random_patterns(content):
    """Fix common numpy random patterns."""
    # Replace np.random.seed() with proper RNG initialization
    content = re.sub(r"np\.random\.seed\((\d+)\)", r"rng = np.random.default_rng(seed=\1)", content)

    # Replace other seed patterns
    content = re.sub(r"np\.random\.seed\(([^)]+)\)", r"rng = np.random.default_rng(seed=\1)", content)

    # Replace specific functions
    replacements = {
        "np.random.randn(": "rng.standard_normal(",
        "np.random.normal(": "rng.normal(",
        "np.random.uniform(": "rng.uniform(",
        "np.random.randint(": "rng.integers(",
        "np.random.random(": "rng.random(",
        "np.random.choice(": "rng.choice(",
        "np.random.shuffle(": "rng.shuffle(",
        "np.random.permutation(": "rng.permutation(",
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content


def ensure_rng_available(content, filename):
    """Ensure RNG is available in test functions."""
    lines = content.split("\n")
    new_lines = []

    # Track if we're inside a test function or fixture
    in_test_function = False
    function_indent = 0
    has_rng_param = False
    needs_rng = False

    for i, line in enumerate(lines):
        # Check if we're entering a test function or fixture
        if re.match(r"^\s*(def test_|@pytest\.fixture)", line):
            in_test_function = True
            function_indent = len(line) - len(line.lstrip())
            has_rng_param = "rng" in line or "def sample_csv_data(self, tmp_path)" in line
            needs_rng = False

        # Check if we're leaving the function
        elif in_test_function and line.strip() and len(line) - len(line.lstrip()) <= function_indent:
            if line.strip().startswith(("def ", "class ", "@")):
                in_test_function = False

        # Check if line uses rng
        if "rng." in line and in_test_function:
            needs_rng = True

        # If we're in a function that needs RNG but doesn't have it as parameter
        if in_test_function and needs_rng and not has_rng_param and line.strip().startswith("def ") and "test_" in line:
            # Add rng parameter to test function
            if "(self)" in line:
                line = line.replace("(self)", "(self, rng)")
            elif "(self," in line:
                # Add rng to existing parameters
                line = line.replace("(self,", "(self, rng,")
            has_rng_param = True

        # If we need RNG but don't have it as param, create local RNG
        if (
            in_test_function
            and needs_rng
            and not has_rng_param
            and "rng." in line
            and "rng = np.random.default_rng" not in line
        ):
            # Insert RNG creation before the first use
            indent = " " * (len(line) - len(line.lstrip()))
            rng_line = f"{indent}rng = np.random.default_rng(seed=42)"
            new_lines.append(rng_line)
            has_rng_param = True  # Prevent multiple insertions

        new_lines.append(line)

    return "\n".join(new_lines)


def fix_file(filepath):
    """Fix a single test file."""
    print(f"Fixing {filepath}")

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Apply numpy random fixes
    content = fix_numpy_random_patterns(content)

    # Ensure RNG is available where needed
    content = ensure_rng_available(content, filepath)

    # Write back if changed
    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✓ Updated {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False


def main():
    """Main function to fix all test files."""
    print("🔧 Fixing numpy.random violations in test files...")

    # Get all test files
    test_dir = Path("tests")
    test_files = list(test_dir.glob("test_*.py"))

    # Add conftest.py (already fixed but double-check)
    test_files.append(test_dir / "conftest.py")

    fixed_files = 0

    for test_file in sorted(test_files):
        if test_file.exists() and fix_file(test_file):
            fixed_files += 1

    print(f"\n✅ Fixed {fixed_files} files")

    # Check remaining violations
    print("\n📊 Checking remaining NPY002 violations...")
    stdout, stderr, code = run_command("uv run ruff check tests/ --select NPY002")

    if code == 0:
        print("🎉 All NPY002 violations fixed!")
    else:
        violations = get_npy002_violations()
        print(f"⚠️  Still {len(violations)} violations remaining:")
        for v in violations[:10]:  # Show first 10
            print(f"  {v['filename']}:{v['location']['row']} - {v['message']}")


if __name__ == "__main__":
    main()
