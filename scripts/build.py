"""
Build a runnable notebook from a jupytext py:percent dev source file.

Usage: build("notebooks/dev_CA1.py") -> build/CA1.ipynb
"""
import base64
import logging
import mimetypes
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

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

LOAD_CLEAN_RE = re.compile(r"^\s*#?\s*%%load_clean\b")
LOAD_EXT_RE = re.compile(r"^\s*#?\s*%load_ext\b")
IMPORT_RE = re.compile(r"^\s*(import|from)\s")
IMPORTS_MARKER_RE = re.compile(r"^\s*#\s*<\$IMPORTS>\s*$")
SYNTHETIC_LIST_RE = re.compile(r"^\s*_LOAD_CLEAN_IMPORTS_\w+\s*=\s*\[")
SYNTHETIC_LIST_ITEM_RE = re.compile(r"^\s*(\w+),?\s*$")
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<path>(?!attachment:)(?!https?://)[^)\s]+)\)"
)


class BuildError(Exception):
    """Raised when a build cannot complete due to unresolved dependencies or missing assets."""


def is_load_ext_cell(source: str) -> bool:
    """
    Detects `%load_ext` invocations.
    @param source - Cell source text.
    @returns True if any line in the cell invokes %load_ext.
    """
    return any(LOAD_EXT_RE.match(line) for line in source.splitlines())


def is_imports_cell(source: str) -> bool:
    """
    Detects the designated imports cell.
    @param source - Cell source text.
    @returns True if the cell contains the `# <$IMPORTS>` marker.
    """
    return any(IMPORTS_MARKER_RE.match(line) for line in source.splitlines())


def extract_imports_cell(source: str) -> str:
    """
    Extracts import lines from an imports cell.
    @param source - Imports cell source text.
    @returns Cell source with the marker line stripped.
    """
    lines = [line for line in source.splitlines() if not IMPORTS_MARKER_RE.match(line)]
    return "\n".join(lines).strip()


def strip_load_clean_cell(source: str) -> str:
    """
    Removes the %%load_clean marker, import line, and synthetic list, keeping only the module body.
    Used as a fallback when a %%load_clean cell can't be sliced (e.g. due to an error).
    @param source - %%load_clean cell source text.
    @returns The cell body with directives stripped.
    """
    lines = source.splitlines()
    if not lines or not LOAD_CLEAN_RE.match(lines[0]):
        return source

    lines = lines[1:]

    while lines and (IMPORT_RE.match(lines[0]) or not lines[0].strip()):
        lines.pop(0)

    if lines and SYNTHETIC_LIST_RE.match(lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip().endswith("]"):
            lines.pop(0)
        if lines:
            lines.pop(0)

    return "\n".join(lines).strip()


def parse_synthetic_list(source: str) -> Set[str]:
    """
    Parses a synthetic `_LOAD_CLEAN_IMPORTS_* = [name1, name2, ...]` list.
    @param source - Cell source text containing the synthetic list.
    @returns Set of target names declared in the list.
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
    """
    Detects a synthetic imports list.
    @param source - Cell source text.
    @returns True if the cell contains a `_LOAD_CLEAN_IMPORTS_*` list.
    """
    return any(SYNTHETIC_LIST_RE.match(line) for line in source.splitlines())


# --- Import cell handling ---------------------------------------------------

def find_imports_cell_index(cells: List) -> Optional[int]:
    """
    Locates the imports cell.
    @param cells - Notebook cell list.
    @returns Index of the imports cell, or None if absent.
    """
    for i, cell in enumerate(cells):
        if is_imports_cell(cell.source):
            return i
    return None


def parse_imports(source: str) -> Dict[str, str]:
    """
    Parses import source into a name -> import-line lookup.
    @param source - Imports cell source text.
    @returns Dict mapping each imported name to the full import line that provides it.
    """
    imports: Dict[str, str] = {}
    for line in source.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        module_path, targets = parse_import_line(line)
        if module_path is None:
            continue

        if not targets:
            imports[module_path.split(".")[-1]] = line
        else:
            for target in targets:
                imports[target] = line
    return imports


def group_imports(import_lines: List[str]) -> List[str]:
    """
    Groups and deduplicates import lines, combining same-module `from` imports.
    @param import_lines - Raw, deduplicated import lines.
    @returns Grouped import lines: plain imports first (sorted), then grouped `from` imports (sorted by module).
    """
    import ast

    from_imports: Dict[str, Set[str]] = {}
    plain_imports: Set[str] = set()

    for line in import_lines:
        line = line.strip()
        if not line:
            continue

        try:
            node = ast.parse(line, mode="exec").body[0]
        except (SyntaxError, IndexError):
            plain_imports.add(line)
            continue

        if isinstance(node, ast.Import):
            plain_imports.add(line)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = {
                f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                for alias in node.names
            }
            from_imports.setdefault(node.module, set()).update(names)
        else:
            plain_imports.add(line)

    result = sorted(plain_imports)

    for module in sorted(from_imports.keys()):
        names = sorted(from_imports[module])
        if "*" in names and len(names) > 1:
            names.remove("*")
        result.append(f"from {module} import {', '.join(names)}")

    return result


def deduplicate_imports(import_lines: Set[str]) -> List[str]:
    """
    Deduplicates, groups, and sorts import lines.
    @param import_lines - Raw import lines to consolidate.
    @returns Final grouped import lines ready to write into the imports cell.
    """
    seen: Set[str] = set()
    unique_lines = []
    for line in sorted(import_lines):
        normalized = line.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_lines.append(normalized)

    return group_imports(unique_lines)


# --- %%load_clean expansion (explicit targets only) -------------------------

def expand_load_clean_cell(
    source: str,
    available_imports: Dict[str, str],
    module_imports_cache: Dict[str, Tuple[object, str]],
    collected_imports: Set[str],
    defined_names: Set[str],
) -> Tuple[str, List[str], List[str]]:
    """
    Expands a %%load_clean cell into self-contained source, using only the explicitly
    requested target names -- no automatic same-module transitive dependency pulling.
    Any internal reference the target needs that isn't already available is reported
    as an error and the cell falls back to `strip_load_clean_cell`, so the build can
    report all problems in one pass rather than stopping at the first one.
    @param source - %%load_clean cell source text.
    @param available_imports - Name -> import-line lookup from the notebook's imports cell.
    @param module_imports_cache - Cache of resolved (result, module_source) keyed by module+targets.
    @param collected_imports - Mutated in place: accumulates import lines needed by expanded cells.
    @param defined_names - Names already available from the import cell and earlier expanded
        cells in this build pass. Mutated in place: successfully sliced names are added.
    @returns Tuple of (expanded_source, warnings, errors).
    """
    warnings: List[str] = []
    errors: List[str] = []

    lines = source.splitlines()
    if not lines or not LOAD_CLEAN_RE.match(lines[0]):
        return source, warnings, errors

    import_line = ""
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if IMPORT_RE.match(stripped):
            import_line = stripped
        break

    if not import_line:
        return strip_load_clean_cell(source), warnings, errors

    module_path, target_names = parse_import_line(import_line)
    if module_path is None:
        return strip_load_clean_cell(source), warnings, errors

    if has_synthetic_list(source):
        synthetic_targets = parse_synthetic_list(source)
        if synthetic_targets:
            target_names = synthetic_targets

    if target_names == {"*"}:
        errors.append(
            f"  [ERROR] {module_path}: wildcard import has no synthetic "
            f"_LOAD_CLEAN_IMPORTS_* list to expand -- re-run the %%load_clean "
            f"cell in the notebook first, then re-sync/build."
        )
        return strip_load_clean_cell(source), warnings, errors

    cache_key = f"{module_path}:{sorted(target_names)}"
    if cache_key in module_imports_cache:
        result, module_source = module_imports_cache[cache_key]
    else:
        filepath = resolve_module_path(module_path)
        if filepath is None:
            errors.append(f"  [ERROR] Module not found: {module_path}")
            return strip_load_clean_cell(source), warnings, errors

        try:
            module_source = filepath.read_text(encoding="utf-8")
        except OSError as e:
            errors.append(f"  [ERROR] Could not read {filepath}: {e}")
            return strip_load_clean_cell(source), warnings, errors

        result = DependencyResolver(module_source, target_names).resolve()
        module_imports_cache[cache_key] = (result, module_source)

    body = DependencyResolver(module_source, target_names).slice_source(target_names)

    still_missing = result.internal_refs - target_names - defined_names
    for ref in sorted(still_missing):
        errors.append(
            f"  [ERROR] '{ref}' (needed by {module_path}) is not in the import line "
            f"and not already defined by an earlier cell or the import cell -- "
            f"add it explicitly, e.g.:\n"
            f"          from {module_path} import {', '.join(sorted(target_names | still_missing))}"
        )

    if errors:
        return strip_load_clean_cell(source), warnings, errors

    defined_names.update(target_names)

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


# --- Jupytext pair sync ------------------------------------------------------

def find_paired_ipynb(src_path: str) -> Optional[Path]:
    """
    Locates the paired .ipynb for a dev_*.py source.
    @param src_path - Path to the .py source file.
    @returns The paired .ipynb path, or None if it doesn't exist.
    """
    candidate = Path(src_path).with_suffix(".ipynb")
    return candidate if candidate.exists() else None


def sync_jupytext_pair(src_path: str) -> None:
    """
    Runs jupytext's own sync so the paired .ipynb and .py are reconciled before either is read.
    No-op if no paired .ipynb exists.
    @param src_path - Path to the .py source file.
    """
    if find_paired_ipynb(src_path) is None:
        return

    jupytext_cli(["--sync", src_path])


# --- Markdown image baking ---------------------------------------------------

def bake_markdown_images(notebook: nbformat.NotebookNode, src_path: str) -> List[str]:
    """
    Embeds local markdown image references as base64 cell attachments, making the
    built notebook self-contained. Rewrites `![alt](relative/path)` references to
    `![alt](attachment:filename)`. References already using `attachment:` or an
    `http(s)://` URL are left untouched.
    @param notebook - Notebook to mutate in place.
    @param src_path - Path to the .py source file; image paths resolve relative to its directory.
    @returns List of error strings for missing files or undetectable MIME types.
    """
    errors: List[str] = []
    base_dir = Path(src_path).parent

    for cell in notebook.cells:
        if cell.cell_type != "markdown" or not cell.source:
            continue

        def replace(match: re.Match) -> str:
            alt = match.group("alt")
            rel_path = match.group("path")
            image_path = (base_dir / rel_path).resolve()

            if not image_path.is_file():
                errors.append(f"  [ERROR] Markdown image not found: '{rel_path}' (resolved to {image_path})")
                return match.group(0)

            mime_type, _ = mimetypes.guess_type(image_path.name)
            if mime_type is None or not mime_type.startswith("image/"):
                errors.append(f"  [ERROR] Could not determine image MIME type for '{rel_path}' (resolved to {image_path})")
                return match.group(0)

            data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            attachments = cell.get("attachments") or {}
            attachments[image_path.name] = {mime_type: data}
            cell["attachments"] = attachments

            return f"![{alt}](attachment:{image_path.name})"

        cell.source = MARKDOWN_IMAGE_RE.sub(replace, cell.source)

    return errors


# --- Build --------------------------------------------------------------

def build(src_path: str, out_dir: str = "build") -> Path:
    """
    Builds a runnable notebook from a jupytext py:percent dev source file.
    Syncs the paired .ipynb, bakes local markdown images into attachments, strips
    %%load_clean/%load_ext directives (slicing %%load_clean cells down to exactly
    their requested targets), and merges all collected imports into the imports cell.
    @param src_path - Path to the dev_*.py source file.
    @param out_dir - Directory to write the built .ipynb into.
    @returns Path to the built notebook.
    @throws BuildError if any dependency or asset cannot be resolved.
    """
    sync_jupytext_pair(src_path)

    notebook = jupytext.read(src_path, fmt="py:percent")

    build_errors: List[str] = bake_markdown_images(notebook, src_path)
    for error in build_errors:
        logger.error(error)

    imports_source = ""
    for cell in notebook.cells:
        if is_imports_cell(cell.source):
            imports_source = extract_imports_cell(cell.source)
            break

    available_imports = parse_imports(imports_source)
    module_cache: Dict[str, Tuple[object, str]] = {}
    all_collected_imports: Set[str] = set()
    defined_names: Set[str] = set(available_imports.keys())

    kept_cells = []
    for cell in notebook.cells:
        if is_load_ext_cell(cell.source):
            continue

        if cell.source and LOAD_CLEAN_RE.match(cell.source.splitlines()[0]):
            cell.source, warnings, errors = expand_load_clean_cell(
                cell.source, available_imports, module_cache, all_collected_imports, defined_names
            )
            for warning in warnings:
                logger.warning(warning)
            for error in errors:
                logger.error(error)
            build_errors.extend(errors)

        kept_cells.append(cell)

    if build_errors:
        raise BuildError(
            f"Build failed for {src_path}: {len(build_errors)} error(s). "
            f"See logged [ERROR] messages above."
        )

    imports_idx = find_imports_cell_index(kept_cells)

    if all_collected_imports:
        for line in imports_source.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                all_collected_imports.add(line)

        deduped = deduplicate_imports(all_collected_imports)
        import_cell_content = "# <$IMPORTS>\n" + "\n".join(deduped)

        if imports_idx is not None:
            kept_cells[imports_idx].source = import_cell_content
        else:
            kept_cells.insert(0, nbformat.v4.new_code_cell(import_cell_content))

    notebook.cells = kept_cells

    out_name = Path(src_path).stem.removeprefix("dev_") + ".ipynb"
    out_path = Path(out_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, out_path)

    return out_path


def build_default(notebooks_dir: str = "notebooks", out_dir: str = "build") -> List[Path]:
    """
    Builds every dev_*.py notebook in a directory.
    @param notebooks_dir - Directory to scan for dev_*.py source files.
    @param out_dir - Directory to write built notebooks into.
    @returns Paths to all built notebooks.
    """
    from scripts.sync import sync

    sync(silent=True, start_message="Syncing before build...")

    dev_files = sorted(Path(notebooks_dir).glob("dev_*.py"))

    if not dev_files:
        logger.warning(f"No dev_*.py files found in {notebooks_dir}/")
        return []

    return [build(str(p), out_dir) for p in dev_files]