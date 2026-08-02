"""
Strip %%load_clean magic lines and %load_ext cells from the jupytext
py:percent source, then convert the result to a notebook via jupytext.

Usage: build("notebooks/dev_CA1.py") -> build/CA1.ipynb

Features:
- Detects imports from # <$IMPORTS> marker cells
- Resolves module dependencies for %%load_clean cells
- Parses synthetic _LOAD_CLEAN_IMPORTS_* lists for wildcard target names
- Merges all needed imports into the import cell (deduplicated + grouped)
- Warns about missing internal module dependencies
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jupytext
import nbformat

from dependencies.dependency_resolver import (
    DependencyResolver,
    resolve_module_path,
    parse_import_line,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LOAD_CLEAN_RE = re.compile(r"^\s*#?\s*%%load_clean\b")
LOAD_EXT_RE = re.compile(r"^\s*#?\s*%load_ext\b")
IMPORT_RE = re.compile(r"^\s*(import|from)\s")
IMPORTS_MARKER_RE = re.compile(r"^\s*#\s*<\$IMPORTS>\s*$")
SYNTHETIC_LIST_RE = re.compile(r"^\s*_LOAD_CLEAN_IMPORTS_\w+\s*=\s*\[")
SYNTHETIC_LIST_ITEM_RE = re.compile(r"^\s*(\w+),?\s*$")


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
# %%load_clean expansion with dependency resolution
# ---------------------------------------------------------------------------

def expand_load_clean_cell(
    source: str,
    available_imports: Dict[str, str],
    module_imports_cache: Dict[str, str],
    collected_imports: Set[str],
) -> Tuple[str, List[str]]:
    """
    Expand a %%load_clean cell into self-contained source.
    Collects needed imports into collected_imports set (for merging into import cell later).

    Returns:
        (expanded_source, warnings)
    """
    warnings: List[str] = []
    lines = source.splitlines()
    if not lines or not LOAD_CLEAN_RE.match(lines[0]):
        return source, warnings

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
        return strip_load_clean_cell(source), warnings

    module_path, target_names = parse_import_line(import_line)
    if module_path is None:
        return strip_load_clean_cell(source), warnings

    # Check for synthetic list (only present for wildcard imports)
    if has_synthetic_list(source):
        synthetic_targets = parse_synthetic_list(source)
        if synthetic_targets:
            target_names = synthetic_targets
            logger.info(f"  [INFO] Using synthetic list targets: {', '.join(sorted(synthetic_targets))}")

    # Check cache
    cache_key = f"{module_path}:{sorted(target_names)}"
    if cache_key in module_imports_cache:
        body = module_imports_cache[cache_key]
    else:
        filepath = resolve_module_path(module_path)
        if filepath is None:
            warnings.append(f"  [WARN] Module not found: {module_path}")
            return strip_load_clean_cell(source), warnings

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                module_source = f.read()
        except OSError as e:
            warnings.append(f"  [WARN] Could not read {filepath}: {e}")
            return strip_load_clean_cell(source), warnings

        resolver = DependencyResolver(module_source, target_names)
        result = resolver.resolve()
        body = resolver.slice_source(result.needed)
        module_imports_cache[cache_key] = body

        # Check for missing internal dependencies
        if result.internal_refs:
            missing = result.internal_refs - set(available_imports.keys())
            for ref in sorted(missing):
                warnings.append(
                    f"  [WARN] '{ref}' (needed by {module_path}) is an internal module "
                    f"reference not found in import cell — ensure it's loaded via another %%load_clean"
                )

    # Collect imports needed by this cell
    filepath = resolve_module_path(module_path)
    if filepath:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                module_source = f.read()
            resolver = DependencyResolver(module_source, target_names)
            result = resolver.resolve()

            for imp in result.module_imports:
                if imp.is_external:
                    collected_imports.add(imp.import_line)

            for ref in result.external_refs:
                if ref in available_imports:
                    collected_imports.add(available_imports[ref])
        except Exception:
            pass

    return body, warnings


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(src_path: str, out_dir: str = "build") -> Path:
    logger.info(f"Building {src_path}...")
    notebook = jupytext.read(src_path, fmt="py:percent")

    # First pass: collect imports from import cell (using original notebook)
    imports_source = ""
    for cell in notebook.cells:
        if is_imports_cell(cell.source):
            imports_source = extract_imports_cell(cell.source)
            logger.info(f"  [INFO] Import cell found in original notebook")
            break

    available_imports = parse_imports(imports_source)
    if available_imports:
        logger.info(f"  [INFO] Available imports: {', '.join(sorted(available_imports.keys()))}")

    # Cache for module bodies
    module_cache: Dict[str, str] = {}

    # Collect all imports needed by expanded cells
    all_collected_imports: Set[str] = set()

    # Second pass: expand cells and track which to keep
    kept_cells = []
    for i, cell in enumerate(notebook.cells):
        # Skip load_ext cells
        if is_load_ext_cell(cell.source):
            logger.info(f"  [INFO] Skipping %load_ext cell at index {i}")
            continue

        # Expand load_clean cells
        if LOAD_CLEAN_RE.match(cell.source.splitlines()[0] if cell.source else ""):
            logger.info(f"  [INFO] Expanding %%load_clean cell at index {i}")
            cell.source, warnings = expand_load_clean_cell(
                cell.source, available_imports, module_cache, all_collected_imports
            )
            for w in warnings:
                logger.warning(w)

        kept_cells.append(cell)

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
            logger.info(f"  [INFO] Updated import cell at index {imports_idx} with {len(deduped)} imports")
        else:
            # Create new import cell at the beginning
            new_cell = nbformat.v4.new_code_cell(import_cell_content)
            kept_cells.insert(0, new_cell)
            logger.info(f"  [INFO] Created import cell with {len(deduped)} imports")

    notebook.cells = kept_cells

    out_name = Path(src_path).stem.removeprefix("dev_") + ".ipynb"
    out_path = Path(out_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, out_path)

    logger.info(f"Built {out_path}")
    return out_path


def build_default(notebooks_dir: str = "notebooks", out_dir: str = "build") -> List[Path]:
    from scripts.sync import sync

    sync(silent=True, start_message="Syncing before build...")

    dev_files = sorted(Path(notebooks_dir).glob("dev_*.py"))

    if not dev_files:
        logger.warning(f"No dev_*.py files found in {notebooks_dir}/")
        return []

    return [build(str(p), out_dir) for p in dev_files]
