"""Curated reports and their LaTeX build support."""

import os
from pathlib import Path


def write_analysis_note(root, filename, title, lines, data_file):
    """Write a numerical summary linking to the single maintained narrative."""
    root = Path(root)
    reports = Path(__file__).resolve().parent
    markdown = Path(os.path.relpath(reports / 'PRL_REPRODUCTION.md', root)).as_posix()
    pdf = Path(os.path.relpath(reports / 'PRL_REPRODUCTION.pdf', root)).as_posix()
    header = [f'# {title}', '',
              f'Generated numerical summary of [{data_file}]({data_file}).',
              'This note does not update or validate a published report edition.',
              f'Model definitions, protocols and interpretation: [PRL report]({markdown}) '
              f'([PDF]({pdf})).', '']
    (root / filename).write_text('\n'.join(header + lines) + '\n')
