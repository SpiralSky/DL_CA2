"""
Strip %%load_clean magic lines and %load_ext cells from the jupytext
py:percent source, then convert the result to a notebook via jupytext.

Usage: build("notebooks/dev_CA1.py") -> build/CA1.ipynb

Features:
- Detects imports from # <$IMPORTS> marker cells
- Slices %%load_clean cells down to exactly the requested target names
  (no automatic same-module transitive dependency pulling -- dependencies
  must be declared explicitly on the import line)
- Parses synthetic _LOAD_CLEAN_IMPORTS_* lists for wildcard target names
- Merges all needed imports into the import cell (deduplicated + grouped)
- Errors (fails the build) on missing internal module dependencies;
  warns on missing external package dependencies
- Syncs the .py source with its paired .ipynb before reading, then carries
  over cell attachments (e.g. pasted images in markdown cells) from the
  .ipynb, since jupytext's py:percent format has no text representation
  for attachments and would otherwise silently drop them
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jupytext
import nbformat
from jupytext.cli import jupytext as jupytext_cli

from dependencies.dependency_resolver import (
    DependencyResolver,
    resolve_module_path,
    parse_import_line,
)

# Logging is reserved for problems only (warnings/errors) -- no [INFO] noise.
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

LOAD_CLEAN_RE = re.compile(r"^\s*#?\s*%%load_clean\b")
LOAD_EXT_RE = re.compile(r"^\s*#?\s*%load_ext\b")
IMPORT_RE = re.compile(r"^\s*(import|from)\s")
IMPORTS_MARKER_RE = re.compile(r"^\s*#\s*<\$IMPORTS>\s*$")
SYNTHETIC_LIST_RE = re.compile(r"^\s*_LOAD_CLEAN_IMPORTS_\w+\s*=\s*\[")
SYNTHETIC_LIST_ITEM_RE = re.compile(r"^\s*(\w+),?\s*$")


class BuildError(Exception):
    """Raised when a build cannot complete due to unresolved dependencies."""


def is_load_ext_cell(source: str) -> bool:
    """True if any line in the cell invokes %load_ext."""
    return any(LOAD_EXT_RE.match(line) for line in source.splitlines())


def is_imports_cell(source: str) -> bool:
    """True if cell contains the # <$IMPORTS> marker."""
    return any(IMPORTS_MARKER_RE.match(line) for line in source.splitlines())


def extract_imports_cell(source: str) -> str:
    """Extract import lines from an imports cell (strip the marker line)."""
    lines = source.splitlines()
    filtered = []
    for line in lines:
        if IMPORTS_MARKER_RE.match(line):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def strip_load_clean_cell(source: str) -> str:
    """
    Remove the %%load_clean marker, import line, and synthetic list.
    Keep only the actual module body.
    """
    lines = source.splitlines()
    if not lines or not LOAD_CLEAN_RE.match(lines[0]):
        return source

    lines = lines[1:]  # drop marker

    # Drop import line
    while lines and (IMPORT_RE.match(lines[0]) or not lines[0].strip()):
        lines.pop(0)

    # Drop synthetic list
    if lines and SYNTHETIC_LIST_RE.match(lines[0]):
        lines.pop(0)  # drop opening line
        while lines and not lines[0].strip().endswith("]"):
            lines.pop(0)
        if lines:
            lines.pop(0)  # drop closing ]

    return "\n".join(lines).strip()


def parse_synthetic_list(source: str) -> Set[str]:
    """
    Parse _LOAD_CLEAN_IMPORTS_* = [name1, name2, ...]
    to extract the set of target names.
    """
    targets = set()
    in_list = False
    for line in source.splitlines():
        if SYNTHETIC_LIST_RE.match(line):
            in_list = True
            continue
        if in_list:
            if line.strip().endswith("]"):
                break
            match = SYNTHETIC_LIST_ITEM_RE.match(line)
            if match:
                targets.add(match.group(1))
    return targets


def has_synthetic_list(source: str) -> bool:
    """True if cell contains a synthetic _LOAD_CLEAN_IMPORTS list."""
    return any(SYNTHETIC_LIST_RE.match(line) for line in source.splitlines())


# ---------------------------------------------------------------------------
# Import cell handling
# ---------------------------------------------------------------------------

def find_imports_cell_index(cells: List) -> Optional[int]:
    """Find the index of the imports cell in the given cell list."""
    for i, cell in enumerate(cells):
        if is_imports_cell(cell.source):
            return i
    return None


def parse_imports(source: str) -> Dict[str, str]:
    """Parse import source into a dict: {imported_name -> full_import_line}."""
    imports: Dict[str, str] = {}
    for line in source.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        module_path, targets = parse_import_line(line)
        if module_path is None:
            continue

        if not targets:
            name = module_path.split(".")[-1]
            imports[name] = line
        else:
            for t in targets:
                imports[t] = line
    return imports


def group_imports(import_lines: List[str]) -> List[str]:
    """
    Group and deduplicate import lines.

    Combines multiple 'from x import y' lines with the same module into
    a single 'from x import a, b, c' line.

    Also groups 'import x' and 'import x as y' separately.
    """
    # Parse all imports
    from_imports: Dict[str, Set[str]] = {}  # module -> set of names
    plain_imports: Set[str] = set()  # "import x" or "import x as y"

    for line in import_lines:
        line = line.strip()
        if not line:
            continue

        try:
            tree = __import__('ast').parse(line, mode='exec')
            node = tree.body[0]
        except (SyntaxError, IndexError):
            plain_imports.add(line)
            continue

        if isinstance(node, __import__('ast').Import):
            # "import x" or "import x as y" or "import x.y"
            plain_imports.add(line)
        elif isinstance(node, __import__('ast').ImportFrom) and node.module:
            # "from x import a, b, c"
            module = node.module
            names = set()
            for alias in node.names:
                if alias.asname:
                    names.add(f"{alias.name} as {alias.asname}")
                else:
                    names.add(alias.name)

            if module not in from_imports:
                from_imports[module] = set()
            from_imports[module].update(names)
        else:
            plain_imports.add(line)

    # Build grouped lines
    result = []

    # Plain imports first (sorted)
    for line in sorted(plain_imports):
        result.append(line)

    # Grouped from imports (sorted by module name)
    for module in sorted(from_imports.keys()):
        names = sorted(from_imports[module])
        # Skip wildcard if there are explicit names
        if "*" in names and len(names) > 1:
            names.remove("*")
        result.append(f"from {module} import {', '.join(names)}")

    return result


def deduplicate_imports(import_lines: Set[str]) -> List[str]:
    """Deduplicate, group, and sort import lines."""
    seen = set()
    unique_lines = []
    for line in sorted(import_lines):
        normalized = line.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_lines.append(normalized)

    return group_imports(unique_lines)


# ---------------------------------------------------------------------------
# %%load_clean expansion (explicit targets only -- no auto dependency pull)
# ---------------------------------------------------------------------------

def expand_load_clean_cell(
    source: str,
    available_imports: Dict[str, str],
    module_imports_cache: Dict[str, Tuple[object, str]],
    collected_imports: Set[str],
    defined_names: Set[str],
) -> Tuple[str, List[str], List[str]]:
    """
    Expand a %%load_clean cell into self-contained source, using ONLY the
    explicitly requested target names -- no same-module transitive
    dependency pulling. Any internal reference the target needs that isn't
    already available (explicit import cell, or an earlier %%load_clean's
    defined_names) is reported as an error and the cell is left un-sliced
    (falls back to strip_load_clean_cell) so the build can report all
    problems in one pass rather than stopping at the first one.

    Args:
        defined_names: names already made available by the import cell
            and by %%load_clean cells expanded earlier in this same build
            pass. Mutated in place: on success, the newly sliced names are
            added so later cells can see them too.

    Returns:
        (expanded_source, warnings, errors)
    """
    warnings: List[str] = []
    errors: List[str] = []

    lines = source.splitlines()
    if not lines or not LOAD_CLEAN_RE.match(lines[0]):
        return source, warnings, errors

    # Extract import line
    import_line = ""
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if IMPORT_RE.match(stripped):
            import_line = stripped
            break
        break

    if not import_line:
        return strip_load_clean_cell(source), warnings, errors

    module_path, target_names = parse_import_line(import_line)
    if module_path is None:
        return strip_load_clean_cell(source), warnings, errors

    # Check for synthetic list (only present for wildcard imports)
    if has_synthetic_list(source):
        synthetic_targets = parse_synthetic_list(source)
        if synthetic_targets:
            target_names = synthetic_targets

    # A literal "*" should never reach slicing -- it means the cell has a
    # wildcard import but no (or a stale) synthetic list to expand it into
    # real names. Re-running the %%load_clean cell in the notebook first
    # regenerates the synthetic list; here we just fail loudly instead of
    # silently slicing a bogus "*" definition into the output.
    if target_names == {"*"}:
        errors.append(
            f"  [ERROR] {module_path}: wildcard import has no synthetic "
            f"_LOAD_CLEAN_IMPORTS_* list to expand -- re-run the %%load_clean "
            f"cell in the notebook first, then re-sync/build."
        )
        return strip_load_clean_cell(source), warnings, errors

    # Resolve (cached by module+targets) but always re-slice against the
    # *current* target_names only -- resolution result is reused just to
    # avoid re-reading/re-parsing the source file and to get refs/imports.
    cache_key = f"{module_path}:{sorted(target_names)}"
    if cache_key in module_imports_cache:
        result, module_source = module_imports_cache[cache_key]
    else:
        filepath = resolve_module_path(module_path)
        if filepath is None:
            errors.append(f"  [ERROR] Module not found: {module_path}")
            return strip_load_clean_cell(source), warnings, errors

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                module_source = f.read()
        except OSError as e:
            errors.append(f"  [ERROR] Could not read {filepath}: {e}")
            return strip_load_clean_cell(source), warnings, errors

        resolver = DependencyResolver(module_source, target_names)
        result = resolver.resolve()
        module_imports_cache[cache_key] = (result, module_source)

    # Slice out exactly what was requested -- nothing more.
    resolver = DependencyResolver(module_source, target_names)
    body = resolver.slice_source(target_names)

    # Anything the target needs internally that isn't explicitly requested
    # here, and isn't already available from an earlier cell/import, is a
    # hard error: the sliced body will NameError at runtime otherwise.
    still_missing = result.internal_refs - target_names - defined_names
    for ref in sorted(still_missing):
        errors.append(
            f"  [ERROR] '{ref}' (needed by {module_path}) is not in the import line "
            f"and not already defined by an earlier cell or the import cell -- "
            f"add it explicitly, e.g.:\n"
            f"          from {module_path} import {', '.join(sorted(target_names | still_missing))}"
        )

    if errors:
        # Don't mark these names as defined / don't collect their imports --
        # the cell is broken until the user fixes the import line.
        return strip_load_clean_cell(source), warnings, errors

    defined_names.update(target_names)

    # Collect imports needed by this cell (external packages the sliced
    # body itself imports, plus any external refs satisfied by the
    # notebook's own import cell).
    for imp in result.module_imports:
        if imp.is_external:
            collected_imports.add(imp.import_line)

    missing_external = set()
    for ref in result.external_refs:
        if ref in available_imports:
            collected_imports.add(available_imports[ref])
        else:
            missing_external.add(ref)

    for ref in sorted(missing_external):
        warnings.append(
            f"  [WARN] '{ref}' (external, needed by {module_path}) not found in the "
            f"import cell -- make sure it's imported/installed."
        )

    return body, warnings, errors


# ---------------------------------------------------------------------------
# Attachment preservation (jupytext's py:percent format cannot carry
# cell.attachments -- e.g. pasted/embedded images in markdown cells -- so
# they must be synced from and copied out of the paired .ipynb).
# ---------------------------------------------------------------------------

def find_paired_ipynb(src_path: str) -> Optional[Path]:
    """Return the paired .ipynb next to a dev_*.py source, if it exists."""
    candidate = Path(src_path).with_suffix(".ipynb")
    return candidate if candidate.exists() else None


def sync_jupytext_pair(src_path: str) -> None:
    """
    Run jupytext's own --sync on src_path so the paired .ipynb is brought
    up to date with the .py (and vice versa) before either is read. This
    guarantees cell count/order match between the two, which attachment
    copying below depends on.
    """
    if find_paired_ipynb(src_path) is None:
        return

    jupytext_cli(["--sync", src_path])


def merge_attachments_from_pair(notebook: nbformat.NotebookNode, src_path: str) -> None:
    """
    Copy cell.attachments from the paired .ipynb onto the freshly-read
    py:percent notebook, matched by index. Must run before any cell
    filtering/expansion so indices still align, and after
    sync_jupytext_pair() so counts are guaranteed to match.
    """
    pair_path = find_paired_ipynb(src_path)
    if pair_path is None:
        return

    paired = nbformat.read(pair_path, as_version=4)

    if len(paired.cells) != len(notebook.cells):
        raise BuildError(
            f"Build failed for {src_path}: paired .ipynb has "
            f"{len(paired.cells)} cells but .py has {len(notebook.cells)} "
            f"after sync -- pair is out of sync, re-sync manually and retry."
        )

    for src_cell, paired_cell in zip(notebook.cells, paired.cells):
        attachments = paired_cell.get("attachments")
        if attachments:
            src_cell["attachments"] = attachments


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(src_path: str, out_dir: str = "build") -> Path:
    sync_jupytext_pair(src_path)

    notebook = jupytext.read(src_path, fmt="py:percent")
    merge_attachments_from_pair(notebook, src_path)

    # First pass: collect imports from import cell (using original notebook)
    imports_source = ""
    for cell in notebook.cells:
        if is_imports_cell(cell.source):
            imports_source = extract_imports_cell(cell.source)
            break

    available_imports = parse_imports(imports_source)

    # Cache for module resolution results (result, module_source), keyed by
    # module_path+target_names so we don't re-read/re-parse repeatedly.
    module_cache: Dict[str, Tuple[object, str]] = {}

    # Collect all imports needed by expanded cells
    all_collected_imports: Set[str] = set()

    # Names already available at the point we're expanding a given cell:
    # seeded with whatever the import cell explicitly provides, then grown
    # as each %%load_clean cell is successfully expanded, in cell order.
    defined_names: Set[str] = set(available_imports.keys())

    build_errors: List[str] = []

    # Second pass: expand cells and track which to keep
    kept_cells = []
    for cell in notebook.cells:
        # Skip load_ext cells
        if is_load_ext_cell(cell.source):
            continue

        # Expand load_clean cells
        if cell.source and LOAD_CLEAN_RE.match(cell.source.splitlines()[0]):
            cell.source, warnings, errors = expand_load_clean_cell(
                cell.source, available_imports, module_cache, all_collected_imports, defined_names
            )
            for w in warnings:
                logger.warning(w)
            for e in errors:
                logger.error(e)
            build_errors.extend(errors)

        kept_cells.append(cell)

    if build_errors:
        raise BuildError(
            f"Build failed for {src_path}: {len(build_errors)} unresolved dependency "
            f"error(s). See logged [ERROR] messages above."
        )

    # Find imports cell in kept_cells (indices shifted after skipping)
    imports_idx = find_imports_cell_index(kept_cells)

    # Merge collected imports into import cell
    if all_collected_imports:
        # Add existing imports from import cell
        for line in imports_source.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                all_collected_imports.add(line)

        deduped = deduplicate_imports(all_collected_imports)
        import_cell_content = "# <$IMPORTS>\n" + "\n".join(deduped)

        if imports_idx is not None:
            # Update existing import cell
            kept_cells[imports_idx].source = import_cell_content
        else:
            # Create new import cell at the beginning
            new_cell = nbformat.v4.new_code_cell(import_cell_content)
            kept_cells.insert(0, new_cell)

    notebook.cells = kept_cells

    out_name = Path(src_path).stem.removeprefix("dev_") + ".ipynb"
    out_path = Path(out_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, out_path)

    return out_path


def build_default(notebooks_dir: str = "notebooks", out_dir: str = "build") -> List[Path]:
    from scripts.sync import sync

    sync(silent=True, start_message="Syncing before build...")

    dev_files = sorted(Path(notebooks_dir).glob("dev_*.py"))

    if not dev_files:
        logger.warning(f"No dev_*.py files found in {notebooks_dir}/")
        return []

    return [build(str(p), out_dir) for p in dev_files]