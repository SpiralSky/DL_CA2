import os
import sys
import importlib.util
import ast_comments as ast_c
from IPython import get_ipython
from IPython.core.magic import Magics, magics_class, cell_magic

MAX_BLANK_LINES = 2


class _SourceSlicer:
    """Extracts top-level definitions/assignments from source by line
    range instead of rewriting the AST and unparsing it.

    Because the original text is sliced rather than regenerated,
    original formatting (spacing, method chains, inline comments,
    docstrings) survives untouched. Docstrings are kept by default;
    pass keep_docstrings=False to strip them.
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
        """Return (start_line, end_line) of a node's docstring, if any."""
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
        """Count blank lines immediately above the node, capped at MAX_BLANK_LINES."""
        start = self._effective_start_line(node)
        count = 0
        idx = start - 2  # 0-indexed line just above the node's first line
        while idx >= 0 and self.lines[idx].strip() == "" and count < MAX_BLANK_LINES:
            count += 1
            idx -= 1
        return count

    def _leading_standalone_comment(self, body, node):
        """If a standalone comment sits directly above the node with no
        blank line separating them, return its source text so it travels
        with the node it documents.
        """
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

        Re-running the cell re-reads the source file and rewrites the cell
        body with fresh cleaned code, while keeping this header + import
        line intact so the cell keeps working next time you run it.

        Original formatting, comments, and docstrings are preserved
        because the target code is sliced directly out of the source
        file rather than rebuilt from the AST.
        """
        import_line = line.strip()
        if not import_line:
            import_line, _, _ = cell.partition("\n")
            import_line = import_line.strip()

        module_path, target_names = self._parse_import_line(import_line)
        if module_path is None:
            print(f"Could not parse import statement: {import_line!r}")
            return

        filepath = self._resolve_filepath(module_path)
        if not filepath:
            print(f"Module not found: {module_path}")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            clean_code = _SourceSlicer(source, target_names).build()
        except SyntaxError as e:
            print(f"Syntax error in target file {filepath}: {e}")
            return

        output = f"%%load_clean\n{import_line}\n\n{clean_code}"
        get_ipython().set_next_input(output, replace=True)

        exec(compile(clean_code, filepath, "exec"), get_ipython().user_ns)

    @staticmethod
    def _parse_import_line(import_line):
        """Parse a literal `import x.y.z` or `from x.y.z import a, b as c` line."""
        try:
            node = ast_c.parse(import_line, mode="exec").body[0]
        except (SyntaxError, IndexError):
            return None, set()

        if isinstance(node, ast_c.Import) and node.names:
            return node.names[0].name, set()
        if isinstance(node, ast_c.ImportFrom) and node.module:
            targets = {alias.name for alias in node.names}
            return node.module, targets
        return None, set()

    @staticmethod
    def _resolve_filepath(module_path):
        filepath = None
        try:
            spec = importlib.util.find_spec(module_path)
            if spec and spec.origin:
                filepath = spec.origin
        except (ModuleNotFoundError, ValueError):
            pass
        if not filepath:
            fallback_path = module_path.replace(".", "/") + ".py"
            if os.path.exists(fallback_path):
                filepath = fallback_path
        if filepath and os.path.exists(filepath):
            return filepath
        return None


def load_ipython_extension(ipython):
    ipython.register_magics(CustomMagics)