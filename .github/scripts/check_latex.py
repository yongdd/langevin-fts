#!/usr/bin/env python3
"""Check the math in the repository's Markdown files.

GitHub renders math with MathJax, but its Markdown parser runs over the text
first. Inside $...$ and $$...$$ a pair of '*' or '_' becomes emphasis,
'[a](b)' becomes a link and a backslash before punctuation is dropped
('\\,' -> ',', '\\{' -> '{', '\\\\' -> '\\'), so MathJax receives a different
formula or reports "Extra close brace or missing open brace". GitHub passes
the content of two forms through unchanged, and the documentation uses only
those:

    inline:   $`a^{*} \\, b`$

    display:  ```math
              a^{*} \\, b
              ```

This script reports
  1. math written as $...$ or $$...$$, and any other '$' outside code
     (write a literal dollar sign as \\$);
  2. ```math blocks inside a list item: GitHub leaves them as code blocks when
     the list also contains inline math, so the block goes after the item,
     unindented (a numbered list continues with "2." as before);
  3. '|' in inline math in a table row: GitHub ends the cell there even inside
     the code span, so write \\vert (and \\Vert for \\|);
  4. formulas KaTeX cannot parse, if the KaTeX CLI is installed
     (npm install -g katex). GitHub uses MathJax; KaTeX understands nearly
     the same TeX and is the stricter of the two.

Usage:
    python .github/scripts/check_latex.py            # all tracked *.md outside devel/
    python .github/scripts/check_latex.py FILE...    # only these files
"""

import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# A fence line: the container prefix (blockquote markers, list markers,
# indentation), then ``` or ~~~ and the info string.
FENCE = re.compile(r'^((?:[ \t]*(?:>|(?:[-*+]|\d{1,9}[.)])(?=[ \t])))*[ \t]*)(`{3,}|~{3,})(.*)$')
QUOTES = re.compile(r'^(?:[ \t]*>)*')
PREFIX = re.compile(r'^(?:[ \t]*> ?)*[ \t]*')
TABLE_ROW = re.compile(r'^(?:[ \t]*>)*[ \t]*\|')


def scan(text):
    """Return (formulas, problems) of one Markdown text.

    formulas: (line, tex, display) for every $`...`$ and ```math block.
    problems: (line, message).
    """
    formulas, problems = [], []
    fence = None  # (marker, start line, is_math, content lines, blockquote depth)
    for ln, line in enumerate(text.split('\n'), 1):
        m = FENCE.match(line)
        if m and m.group(2)[0] == '`' and '`' in m.group(3):
            m = None  # a backtick fence's info string has no backtick: inline code
        if fence is not None:
            marker, start, is_math, body, depth = fence
            if QUOTES.match(line).group(0).count('>') < depth:
                # the blockquote holding the block ended, and the block with it;
                # this line is ordinary text again
                if is_math:
                    formulas.append((start, '\n'.join(body), True))
                fence = None
            else:
                if m and m.group(2)[0] == marker[0] and len(m.group(2)) >= len(marker) \
                        and not m.group(3).strip() and not re.search(r'[-*+\d]', m.group(1)):
                    if is_math:
                        formulas.append((start, '\n'.join(body), True))
                    fence = None
                elif is_math:
                    body.append(PREFIX.sub('', line, count=1))
                continue
        if m:
            info = m.group(3).split()
            is_math = bool(info) and info[0] == 'math'
            if is_math and not re.fullmatch(r'(?:> ?)*', m.group(1)):
                problems.append((ln, '```math block inside a list item or indented: GitHub may '
                                     'show it as code; put it after the item, unindented'))
            fence = (m.group(2), ln, is_math, [], m.group(1).count('>'))
            continue

        i, n = 0, len(line)
        while i < n:
            c = line[i]
            if c == '\\':
                i += 2
            elif c == '`':
                j = i
                while j < n and line[j] == '`':
                    j += 1
                close = re.compile(r'(?<!`)' + line[i:j] + r'(?!`)').search(line, j)
                i = close.end() if close else j
            elif c == '$':
                if line.startswith('$$', i):
                    problems.append((ln, '$$...$$ math: use a ```math block'))
                    end = line.find('$$', i + 2)
                    i = n if end == -1 else end + 2
                    continue
                if i + 1 < n and line[i + 1] == '`':
                    j = i + 1
                    while j < n and line[j] == '`':
                        j += 1
                    ticks = line[i + 1:j]
                    close = re.compile(r'(?<!`)' + ticks + r'(?!`)').search(line, j)
                    if close and close.end() < n and line[close.end()] == '$':
                        tex = line[j:close.start()]
                        if '|' in tex and TABLE_ROW.match(line):
                            problems.append((ln, "'|' in math inside a table row ends the cell on "
                                                 "GitHub: write \\vert (\\Vert for \\|)"))
                        formulas.append((ln, tex, False))
                        i = close.end() + 1
                        continue
                    problems.append((ln, '$` without a closing `$ on the same line'))
                    i = j
                    continue
                end = line.find('$', i + 1)
                if end == -1:
                    problems.append((ln, "unescaped '$': write a literal dollar sign as \\$"))
                    i += 1
                else:
                    problems.append((ln, '$...$ math: write it as $`...`$ (' + line[i:end + 1][:40] + ')'))
                    i = end + 1
            else:
                i += 1
    if fence is not None:
        problems.append((fence[1], 'code block is not closed'))
    return formulas, problems


def katex_error(katex, tex, display):
    """Return KaTeX's parse error for one formula, or None."""
    args = [katex] + (['--display-mode'] if display else [])
    try:
        r = subprocess.run(args, input=tex, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return 'KaTeX timed out'
    if r.returncode == 0:
        return None
    m = re.search(r'KaTeX parse error: (.+)', r.stderr)
    return m.group(1) if m else r.stderr.strip().split('\n')[0]


def main():
    if len(sys.argv) > 1:
        files = [Path(f).resolve() for f in sys.argv[1:]]
    else:
        listed = subprocess.run(['git', 'ls-files', '*.md'], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.split()
        files = [ROOT / f for f in listed if not f.startswith('devel/')]

    katex = shutil.which('katex')
    jobs, problems = [], []
    for path in files:
        formulas, found = scan(path.read_text(encoding='utf-8'))
        problems += [(path, ln, msg) for ln, msg in found]
        jobs += [(path, ln, tex, display) for ln, tex, display in formulas]

    if katex:
        with ThreadPoolExecutor(max_workers=8) as pool:
            errors = pool.map(lambda j: katex_error(katex, j[2], j[3]), jobs)
            problems += [(p, ln, 'KaTeX: ' + e + ' (' + tex[:50].replace('\n', ' ') + ')')
                         for (p, ln, tex, _), e in zip(jobs, errors) if e]

    for path, ln, msg in sorted(problems, key=lambda p: (str(p[0]), p[1])):
        try:
            name = path.relative_to(ROOT)
        except ValueError:
            name = path
        print(f'{name}:{ln}: {msg}')

    checked = f'{len(jobs)} formulas in {len(files)} files'
    if not katex:
        print('Note: KaTeX CLI not found (npm install -g katex); the TeX syntax was not checked.')
    if problems:
        print(f'{len(problems)} problem(s), {checked}')
        return 1
    print(f'OK: {checked}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
