"""
Strip %%load_clean magic lines and %load_ext cells from the jupytext
py:percent source, then convert the result to a notebook via jupytext.

Usage: build("notebooks/dev_CA1.py") -> build/dev_CA1.ipynb
"""
import re
from pathlib import Path

import jupytext
import nbformat

from scripts.sync import sync

LOAD_CLEAN_RE = re.compile(r"^\s*#?\s*%%load_clean\b")
LOAD_EXT_RE = re.compile(r"^\s*#?\s*%load_ext\b")
IMPORT_RE = re.compile(r"^\s*(import|from)\s")

def is_load_ext_cell(source: str) -> bool:
    """True if any line in the cell invokes %load_ext."""
    return any(LOAD_EXT_RE.match(line) for line in source.splitlines())


def strip_load_clean_cell(source: str) -> str:
    """Remove the `%%load_clean` marker line and the import line(s)
    directly below it, keeping the generated body that follows."""
    lines = source.splitlines()
    if not lines or not LOAD_CLEAN_RE.match(lines[0]):
        return source

    lines = lines[1:]  # drop the marker line
    while lines and (IMPORT_RE.match(lines[0]) or not lines[0].strip()):
        lines.pop(0)  # drop the import line(s) and any blank line(s) after them

    return "\n".join(lines).strip()


def is_jupytext_py(path: Path) -> bool:
    """Heuristic: jupytext py:percent files start with a YAML header
    fenced by '# ---' lines containing 'jupyter:'."""
    with open(path, "r", encoding="utf-8") as f:
        head = "".join(next(f) for _ in range(10) if f)
    return "# ---" in head and "jupyter:" in head


def build(src_path: str, out_dir: str = "build") -> Path:
    print(f"Building {src_path}...")
    notebook = jupytext.read(src_path, fmt="py:percent")

    kept_cells = []
    for cell in notebook.cells:
        if is_load_ext_cell(cell.source):
            continue
        if LOAD_CLEAN_RE.match(cell.source.splitlines()[0] if cell.source else ""):
            cell.source = strip_load_clean_cell(cell.source)
        kept_cells.append(cell)
    notebook.cells = kept_cells

    out_name = Path(src_path).stem.removeprefix("dev_") + ".ipynb"
    out_path = Path(out_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, out_path)

    print(f"Built {out_path}")
    return out_path

def build_default(notebooks_dir: str = "notebooks", out_dir: str = "build") -> list[Path]:
    sync(silent=True, start_message="Syncing before build...")

    dev_files = sorted(Path(notebooks_dir).glob("dev_*.py"))

    if not dev_files:
        print(f"No dev_*.py files found in {notebooks_dir}/")
        return []

    return [build(str(p), out_dir) for p in dev_files]