"""Research workflows; physical models are in the sibling models package.

Use ../workflow.py for supported command-line entry points.
"""


def audit_protected_sources(repo, expected):
    """Check live protected kernels and identify explicitly consolidated runners.

    A historical runner is verified against the pre-cleanup Git revision, not
    mislabeled as byte-identical to its consolidated replacement.
    """
    import hashlib
    import json
    import subprocess
    from pathlib import Path

    package=Path(__file__).resolve().parent
    mapping=json.loads((package/'legacy_commands.json').read_text())
    relocation_file=package.parent/'reports/metadata/relocations.json'
    relocations=json.loads(relocation_file.read_text())['files'] if relocation_file.exists() else []
    relocated={str(package.parent/r['original']):r for r in relocations}
    moved=[];unverified=[]
    for name,digest in expected.items():
        path=repo/name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
            continue
        relocation=relocated.get(str(path))
        if not path.exists() and relocation and relocation['original_sha256']==digest:
            current=package.parent/relocation['current']
            dependencies=relocation.get('dependencies',{})
            dependencies_match=all(
                (package.parent/key).is_file()
                and hashlib.sha256((package.parent/key).read_bytes()).hexdigest()==value
                for key,value in dependencies.items())
            if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest()==relocation['current_sha256'] and dependencies_match:
                moved.append(dict(original=name,baseline=relocation.get('maintenance','2026-09-14 directory relocation'),
                                  original_sha256=digest,current=str(current.relative_to(repo)),
                                  current_sha256=relocation['current_sha256'],
                                  dependencies=dependencies))
                continue
        legacy=mapping.get(path.name)
        if not path.exists() and legacy and path.parent==package.parent:
            result=subprocess.run(['git','show',f'd395ed1a:{name}'],cwd=repo,capture_output=True)
            if result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==digest:
                current=package.parent/(legacy['module'].replace('.','/')+'.py')
                moved.append(dict(original=name,baseline='d395ed1a',original_sha256=digest,
                                  current=str(current.relative_to(repo)),
                                  current_sha256=hashlib.sha256(current.read_bytes()).hexdigest()))
                continue
        unverified.append(name)
    return dict(moved_sources=moved,unverified_sources=unverified)
