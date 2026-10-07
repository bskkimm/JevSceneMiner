"""Check the tracked publication tree; never read local credentials or datasets."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = {'out', 'internal', '.venv', '.superpowers', 'data'}
DATA_SUFFIXES = {'.bag', '.mcap', '.db', '.db3', '.gpkg', '.pcd'}
SECRET = re.compile(rb'\b(?:apikey_[A-Za-z0-9_]{24,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,})\b')


def main():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).split(b'\0')
    errors = []
    for item in paths:
        if not item:
            continue
        rel = Path(item.decode())
        if rel.parts[0] in PRIVATE or rel.suffix in DATA_SUFFIXES or rel.name == '.env':
            errors.append(f'{rel}: local data or credential file is tracked')
            continue
        path = ROOT / rel
        if path.is_file() and SECRET.search(path.read_bytes()):
            errors.append(f'{rel}: credential-like content detected; value withheld')
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
