import hashlib
import re

import ast_comments as ast_c
from IPython import get_ipython
from IPython.core.magic import Magics, magics_class, cell_magic

# Import shared resolver
from dependencies.dependency_resolver import (
    DependencyResolver,
    resolve_module_path,
    parse_import_line,
    normalize_import_line,
    check_dependencies,
)

MAX_BLANK_LINES = 2

# Pattern used to scan previously-executed cell inputs (IPython's `In` list)
# for a prior definition/import of a given name. This lets %%load_clean
# recognize dependencies that were already brought in explicitly by an
# earlier cell (same module or a different one), instead of silently
# re-pulling them itself.
_DEF_PATTERN_TEMPLATE = (
    r"(^\s*(class|def)\s+{name}\b)"
    r"|(\bimport\s+[\w.]*\b{name}\b)"
    r"|(\bfrom\s+[\w.]+\s+import\s+[^#\n]*\b{name}\b)"
    r"|(\bas\s+{name}\b)"
)


def _defined_in_notebook_history(name, ip):
    """True if a previously-executed cell in this session defined or
    imported `name`. Only sees cells that have actually been run, in
    the order they were run (top-to-bottom execution assumed).
    """
    pattern = re.compile(_DEF_PATTERN_TEMPLATE.format(name=re.escape(name)), re.MULTILINE)
    for cell_src in ip.user_ns.get("In", []):
        if cell_src and pattern.search(cell_src):
            return True
    return False


def _dependency_already_available(name, ip):
    """Notebook execution history first (gives us provenance / lets us
    tell the user which cell already provided it), then fall back to
    checking the live namespace (covers names loaded some other way,
    e.g. manually assigned, or from a different module entirely).
    """
    if _defined_in_notebook_history(name, ip):
        return True
    return name in ip.user_ns


class _SourceSlicer:
    """Extracts top-level definitions/assignments from source by line
    range instead of rewriting the AST and unparsing it.
    """

    def __init__(self, source, target_names, keep_docstrings=True):
        self.lines = source.splitlines(keepends=True)
        self.target_names = target_names
        self.keep_docstrings = keep_docstrings
        self.tree = ast_c.parse(source)

    def build(self):
        body = self.tree.body
        keep_nodes = self._select_nodes(body)
        if not keep_nodes:
            return ""

        segments = []
        for i, node in enumerate(keep_nodes):
            leading_comment = self._leading_standalone_comment(body, node)
            blanks = 0 if i == 0 else self._leading_blank_count(node)
            if blanks:
                segments.append("\n" * blanks)
            if leading_comment:
                segments.append(leading_comment)
            segments.append(self._slice_node(node))

        return "".join(segments).strip("\n") + "\n"

    def _select_nodes(self, body):
        if not self.target_names:
            return [n for n in body if not isinstance(n, (ast_c.Import, ast_c.ImportFrom, ast_c.Comment))]

        selected = []
        for node in body:
            if isinstance(node, (ast_c.FunctionDef, ast_c.AsyncFunctionDef, ast_c.ClassDef)):
                if node.name in self.target_names:
                    selected.append(node)
            elif isinstance(node, ast_c.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast_c.Name)]
                if any(t in self.target_names for t in targets):
                    selected.append(node)
        return selected

    def _slice_node(self, node):
        start = self._effective_start_line(node)
        end = node.end_lineno
        docstring_range = self._docstring_line_range(node) if not self.keep_docstrings else None

        raw_lines = self.lines[start - 1:end]
        if docstring_range:
            doc_start, doc_end = docstring_range
            kept = []
            for offset, text_line in enumerate(raw_lines, start=start):
                if doc_start <= offset <= doc_end:
                    continue
                kept.append(text_line)
            raw_lines = kept

        text = "".join(raw_lines)
        if not text.endswith("\n"):
            text += "\n"
        return text

    @staticmethod
    def _effective_start_line(node):
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            return decorators[0].lineno
        return node.lineno

    @staticmethod
    def _docstring_line_range(node):
        body = getattr(node, "body", None)
        if not body:
            return None
        first = body[0]
        if (isinstance(first, ast_c.Expr) and
                isinstance(first.value, ast_c.Constant) and
                isinstance(first.value.value, str)):
            return first.lineno, first.end_lineno
        return None

    def _leading_blank_count(self, node):
        start = self._effective_start_line(node)
        count = 0
        idx = start - 2
        while idx >= 0 and self.lines[idx].strip() == "" and count < MAX_BLANK_LINES:
            count += 1
            idx -= 1
        return count

    def _leading_standalone_comment(self, body, node):
        try:
            pos = body.index(node)
        except ValueError:
            return None
        if pos == 0:
            return None
        prev = body[pos - 1]
        if not isinstance(prev, ast_c.Comment) or prev.inline:
            return None

        start = self._effective_start_line(node)
        if prev.end_lineno != start - 1:
            return None

        comment_line = self.lines[prev.lineno - 1]
        return comment_line if comment_line.endswith("\n") else comment_line + "\n"


@magics_class
class CustomMagics(Magics):

    @cell_magic
    def load_clean(self, line, cell=""):
        """
        %%load_clean
        import path.to.module

        or

        %%load_clean
        from path.to.module import func, ClassName

        Executes the module directly in the current notebook namespace.

        For wildcard imports ("import x.y.z" or "from x import *"),
        generates a synthetic list variable containing all imported names
        for IDE tracking and visibility.

        For explicit imports ("from x import a, b"), no synthetic list
        is generated since names are already explicit.

        Example wildcard output:
          %%load_clean
          from src.models.losses import *  # noqa: F401

          _LOAD_CLEAN_IMPORTS_a7f3 = [
              mse_loss,
              bce_loss,
              perceptual_loss,
          ]

        Example explicit output:
          %%load_clean
          from src.models.training import Trainer, TrainConfig  # noqa: F401

        Dependency checking:
        - Uses scope-aware static analysis (like pyflakes)
        - External package refs: warns if missing
        - Internal module refs: NOT auto-pulled in. If a target depends on
          another name from the same (or a different) module, that name
          must either be explicitly imported/loaded already, or you'll get
          an [ERROR] telling you to add it to the import line yourself.
          "Already loaded" is checked two ways: first against this
          notebook's execution history (previous cells actually run in
          this session), then against the live namespace (covers names
          brought in some other way).
        """
        import_line = line.strip()
        if not import_line:
            import_line, _, _ = cell.partition("\n")
            import_line = import_line.strip()

        module_path, target_names = parse_import_line(import_line)
        if module_path is None:
            print(f"[ERROR] Could not parse import statement: {import_line!r}")
            return

        filepath = resolve_module_path(module_path)
        if not filepath:
            print(f"[ERROR] Module not found: {module_path}")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError as e:
            print(f"[ERROR] Could not read {filepath}: {e}")
            return

        ip = get_ipython()

        # Resolve dependencies (used for reference-analysis only now, not
        # for deciding what gets executed -- no same-module auto-pull).
        resolver = DependencyResolver(source, target_names)
        result = resolver.resolve()

        # Internal refs the target needs that the user didn't explicitly
        # request: these used to be auto-included. Now they're only OK if
        # already available (history or namespace); otherwise it's an error,
        # since running `source` as-is would NameError on exec.
        still_missing = {
            ref for ref in result.internal_refs
            if ref not in target_names and not _dependency_already_available(ref, ip)
        }
        for ref in sorted(still_missing):
            print(
                f"[ERROR] '{ref}' (needed by {module_path}) is not imported and not "
                f"already loaded in this notebook. Add it explicitly, e.g.:\n"
                f"        from {module_path} import {', '.join(sorted(set(target_names or []) | {ref}))}"
            )

        # External package refs still get the normal warning treatment.
        warnings = check_dependencies(
            result.external_refs,
            set(),  # internal refs are handled above; don't double-report them
            ip.user_ns,
            module_path,
            result.module_imports,
        )
        for w in warnings:
            print(w)

        if still_missing:
            print(f"[ERROR] Aborting execution of {module_path}: unresolved dependencies above.")
            return

        # Compile and execute
        try:
            code = compile(source, filepath, "exec")
        except SyntaxError as e:
            print(f"[ERROR] Syntax error in target file {filepath}: {e}")
            return

        exec(code, ip.user_ns)

        # Normalize import line (import x.y.z -> from x.y.z import *)
        normalized = normalize_import_line(import_line)
        if "noqa" not in normalized:
            normalized = normalized + "  # noqa: F401"

        # Generate synthetic import list ONLY for wildcard imports
        # Wildcard = no explicit targets OR explicit "*" target
        is_wildcard = not target_names or target_names == {"*"}

        # For a wildcard, the *actual* names it exposes come from
        # result.needed (everything the resolver found at module top level
        # for this empty/"*" target) -- never from target_names itself,
        # since target_names for a wildcard is either empty or the literal
        # string "*", neither of which is a real name to list.
        wildcard_names = sorted(result.needed) if is_wildcard else []

        synthetic_list = ""
        if wildcard_names:
            cell_hash = hashlib.sha256(normalized.encode()).hexdigest()[:4]
            var_name = f"_LOAD_CLEAN_IMPORTS_{cell_hash}"

            lines = [f"{var_name} = ["]
            lines.extend(f"    {name}," for name in wildcard_names[:-1])
            lines.append(f"    {wildcard_names[-1]}")
            lines.append("]")
            synthetic_list = "\n".join(lines)

        # Build cell output
        if synthetic_list:
            output = f"%%load_clean\n{normalized}\n\n{synthetic_list}"
        else:
            output = f"%%load_clean\n{normalized}"

        ip.set_next_input(output, replace=True)

    @cell_magic
    def load_empty(self, line, cell=""):
        """
        %%load_empty
        import path.to.module

        or

        %%load_empty
        from path.to.module import func, ClassName

        Re-running the cell re-reads the source file and rewrites the cell
        body with fresh cleaned code (imports stripped, only definitions
        kept), while keeping this header + import line intact so the cell
        keeps working next time you run it.

        Note: like %%load_clean, this only slices the explicitly requested
        target_names -- it does not auto-pull other names the target
        depends on from the same module. If the sliced body references a
        name that isn't defined/imported elsewhere, it will NameError when
        executed; add that name to the import line explicitly.
        """
        import_line = line.strip()
        if not import_line:
            import_line, _, _ = cell.partition("\n")
            import_line = import_line.strip()

        module_path, target_names = parse_import_line(import_line)
        if module_path is None:
            print(f"[ERROR] Could not parse import statement: {import_line!r}")
            return

        filepath = resolve_module_path(module_path)
        if not filepath:
            print(f"[ERROR] Module not found: {module_path}")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            clean_code = _SourceSlicer(source, target_names).build()
        except SyntaxError as e:
            print(f"[ERROR] Syntax error in target file {filepath}: {e}")
            return

        if "noqa" not in import_line:
            import_line = import_line + "  # noqa: F401"

        output = f"%%load_empty\n{import_line}\n\n{clean_code}"
        get_ipython().set_next_input(output, replace=True)

        exec(compile(clean_code, filepath, "exec"), get_ipython().user_ns)


def load_ipython_extension(ipython):
    ipython.register_magics(CustomMagics)