"""Build both curated reports through Pandoc and LaTeX/Tectonic.

This typesets frozen results and packaged figures; it never runs simulations.
Edit PRL_REPRODUCTION.md or CLFTS_VALIDATION.md, then run workflow.py reports.
Only replace report PDFs after both builds and text/glyph checks pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from pypdf import PdfReader

REPORTS = ('PRL_REPRODUCTION', 'CLFTS_VALIDATION')


def normalize_table_widths(source):
    """Undo old Pandoc's two-decimal column rounding (six times .17 > 1)."""
    def table(match):
        header = match.group()
        weights = re.findall(r'\\real\{([0-9.]+)\}', header)
        if not weights:
            return header
        total = sum(map(float, weights))
        return re.sub(r'\\real\{([0-9.]+)\}',
                      lambda item: r'\real{' + f'{float(item[1])/total:.12f}' + '}', header)
    return re.sub(r'\\begin\{longtable\}.*?\\toprule', table, source, flags=re.S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-build', action='store_true',
                        help='Keep generated TeX, logs, extracted text and build metadata.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for command in ('pandoc', 'tectonic'):
        if shutil.which(command) is None:
            raise RuntimeError(f'{command} is required; activate the polymerfts environment.')
    build = root / 'build'
    build.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix='edition-', dir=build) as directory:
        stage = Path(directory)
        for name in REPORTS:
            tex = stage / (name + '.tex')
            conversion = subprocess.run([
                'pandoc', str(root / (name + '.md')),
                '--from=markdown+tex_math_single_backslash', '--standalone',
                '--toc', '--toc-depth=2', '--number-sections',
                '--resource-path=' + str(root),
                '--lua-filter=' + str(root / 'layout.lua'),
                '--include-in-header=' + str(root / 'preamble.tex'),
                '--variable=mainfont:DejaVu Sans', '--variable=monofont:DejaVu Sans Mono',
                '--variable=fontsize:10pt', '--variable=geometry:a4paper,margin=20mm',
                '--variable=colorlinks:true', '--variable=urlcolor:blue',
                '--variable=linkcolor:blue', '--variable=linestretch:1.06',
                '--output=' + str(tex),
            ], cwd=root, env={**os.environ, 'CHARGED_REPORT_ROOT': str(root)},
                capture_output=True, text=True)
            (build / (name + '.pandoc.log')).write_text(conversion.stdout + conversion.stderr)
            conversion.check_returncode()
            tex.write_text(normalize_table_widths(tex.read_text()))
            compile_result = subprocess.run([
                'tectonic', '--keep-logs', '--keep-intermediates',
                '--outdir', str(stage), str(tex),
            ], cwd=root, capture_output=True, text=True)
            (build / (name + '.console.log')).write_text(compile_result.stdout + compile_result.stderr)
            # Keep diagnostic sources/logs even if validation fails.
            for suffix in ('.tex', '.log'):
                path = stage / (name + suffix)
                if path.exists():
                    shutil.copy2(path, build / path.name)
            compile_result.check_returncode()
            log = (stage / (name + '.log')).read_text(errors='replace')
            missing = [line for line in log.splitlines() if 'Missing character:' in line]
            overflow = re.findall(r'Overfull \\[hv]box \(([^)]*)\)', log)
            if missing or overflow:
                raise RuntimeError(f'{name}: missing glyphs={missing}, overfull boxes={overflow}; see reports/build logs.')
            pdf = stage / (name + '.pdf')
            reader = PdfReader(pdf)
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
            (build / (name + '.txt')).write_text(text)
            if len(reader.pages) < 5 or len(text) < 10000 or '\ufffd' in text:
                raise RuntimeError(f'{name}: unexpected PDF text or page count.')
            compact_text = re.sub(r'\s+', '', text)
            if 'Executionprotocol' not in compact_text or 'workflow.py' not in compact_text:
                raise RuntimeError(f'{name}: protocol or code map missing from PDF.')
            results[name] = dict(
                pages=len(reader.pages), source_sha256=hashlib.sha256((root / (name + '.md')).read_bytes()).hexdigest(),
                pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                missing_glyphs=0, overfull_boxes=0,
            )
        for name in REPORTS:
            (stage / (name + '.pdf')).replace(root / (name + '.pdf'))
    (build / 'build_summary.json').write_text(json.dumps(results, indent=2) + '\n')
    if not args.keep_build:
        shutil.rmtree(build)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
