import hashlib

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
        - Internal module refs: ERROR if missing
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

        # Resolve dependencies
        resolver = DependencyResolver(source, target_names)
        result = resolver.resolve()

        # Check dependencies
        warnings = check_dependencies(
            result.external_refs,
            result.internal_refs,
            get_ipython().user_ns,
            module_path,
            result.module_imports,
        )
        for w in warnings:
            print(w)

        # Compile and execute
        try:
            code = compile(source, filepath, "exec")
        except SyntaxError as e:
            print(f"[ERROR] Syntax error in target file {filepath}: {e}")
            return

        exec(code, get_ipython().user_ns)

        # Normalize import line (import x.y.z -> from x.y.z import *)
        normalized = normalize_import_line(import_line)
        if "noqa" not in normalized:
            normalized = normalized + "  # noqa: F401"

        # Generate synthetic import list ONLY for wildcard imports
        # Wildcard = no explicit targets OR explicit "*" target
        is_wildcard = not target_names or target_names == {"*"}

        synthetic_list = ""
        if is_wildcard and result.needed:
            cell_hash = hashlib.sha256(normalized.encode()).hexdigest()[:4]
            var_name = f"_LOAD_CLEAN_IMPORTS_{cell_hash}"

            lines = [f"{var_name} = ["]
            for name in sorted(result.needed):
                lines.append(f"    {name},")
            lines.append("]")
            synthetic_list = "\n".join(lines)

        # Build cell output
        if synthetic_list:
            output = f"%%load_clean\n{normalized}\n\n{synthetic_list}"
        else:
            output = f"%%load_clean\n{normalized}"

        get_ipython().set_next_input(output, replace=True)

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
