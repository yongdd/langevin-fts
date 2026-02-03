#!/usr/bin/env python3
"""Check LaTeX syntax in markdown files for GitHub KaTeX compatibility.

GitHub uses KaTeX for math rendering. This script extracts LaTeX blocks
from markdown files and validates them using KaTeX CLI.
"""

import re
import sys
import subprocess
from pathlib import Path

def extract_latex_blocks_with_lines(content):
    """Extract all LaTeX blocks with their line numbers."""
    blocks = []
    lines = content.split('\n')

    # Track display math blocks $$...$$
    in_display_math = False
    display_start = 0
    display_content = []

    for line_num, line in enumerate(lines, 1):
        # Check for $$ delimiter
        if '$$' in line:
            if not in_display_math:
                # Starting a display math block
                in_display_math = True
                display_start = line_num
                # Check if it ends on the same line
                if line.count('$$') >= 2:
                    # Single line display math
                    match = re.search(r'\$\$(.+?)\$\$', line)
                    if match:
                        blocks.append((line_num, match.group(1), 'display'))
                    in_display_math = False
                else:
                    idx = line.index('$$')
                    display_content = [line[idx+2:]]
            else:
                # Ending a display math block
                idx = line.index('$$')
                display_content.append(line[:idx])
                blocks.append((display_start, '\n'.join(display_content), 'display'))
                in_display_math = False
                display_content = []
        elif in_display_math:
            display_content.append(line)

        # Check for inline math $...$ (not $$)
        if not in_display_math:
            inline_matches = re.finditer(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)', line)
            for match in inline_matches:
                blocks.append((line_num, match.group(1), 'inline'))

    return blocks

def check_katex(latex):
    """Check if LaTeX is valid using KaTeX CLI."""
    try:
        result = subprocess.run(
            ['katex'],
            input=latex,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            # Extract error message
            error_lines = result.stderr.strip().split('\n')
            for line in error_lines:
                if 'ParseError' in line:
                    match = re.search(r'KaTeX parse error: (.+)', line)
                    if match:
                        return False, match.group(1)
            return False, "Unknown error"
        return True, None
    except FileNotFoundError:
        return None, "KaTeX CLI not found. Install with: npm install -g katex"
    except subprocess.TimeoutExpired:
        return False, "Timeout"

def check_file(filepath):
    """Check a single markdown file for LaTeX issues."""
    issues = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    latex_blocks = extract_latex_blocks_with_lines(content)

    for line_num, block, block_type in latex_blocks:
        valid, error = check_katex(block)
        if valid is None:
            print(f"Error: {error}")
            sys.exit(1)
        if not valid:
            context = block[:50].replace('\n', ' ')
            issues.append((line_num, error, context))

    return issues

def main():
    """Check all markdown files in docs/."""
    docs_path = Path(__file__).parent.parent / 'docs'

    if not docs_path.exists():
        print(f"Error: {docs_path} not found")
        sys.exit(1)

    total_issues = 0

    for md_file in sorted(docs_path.rglob('*.md')):
        issues = check_file(md_file)
        if issues:
            rel_path = md_file.relative_to(docs_path.parent)
            print(f"\n{rel_path}:")
            for line_num, message, context in issues:
                print(f"  Line {line_num}: {message}")
                if context:
                    print(f"    → {context}...")
            total_issues += len(issues)

    if total_issues == 0:
        print("✓ KaTeX validation passed")
    else:
        print(f"\nFound {total_issues} issue(s)")

    return 1 if total_issues > 0 else 0

if __name__ == '__main__':
    sys.exit(main())
