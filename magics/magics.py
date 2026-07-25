import ast
import importlib.util
import os
import sys
from IPython import get_ipython
from IPython.core.magic import Magics, magics_class, cell_magic


class _CleanTransformer(ast.NodeTransformer):
    """Strips imports/docstrings and optionally filters to specific names."""

    def __init__(self, target_names):
        self.target_names = target_names

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def _strip_docstring(self, node):
        if (node.body and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        self.generic_visit(node)
        return node

    def visit_Module(self, node):
        node = self._strip_docstring(node)
        if self.target_names:
            filtered_body = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if item.name in self.target_names:
                        filtered_body.append(item)
                elif isinstance(item, ast.Assign):
                    targets = [t.id for t in item.targets if isinstance(t, ast.Name)]
                    if any(t in self.target_names for t in targets):
                        filtered_body.append(item)
            node.body = filtered_body
        return node

    def visit_FunctionDef(self, node):
        return self._strip_docstring(node)

    def visit_AsyncFunctionDef(self, node):
        return self._strip_docstring(node)

    def visit_ClassDef(self, node):
        return self._strip_docstring(node)


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
        """
        # allow the import either on the magic's `line` or as the first
        # line of the cell body (so `%%load_clean\nimport x.y.z` also works)
        import_line = line.strip()
        rest = cell
        if not import_line:
            import_line, _, rest = cell.partition("\n")
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
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error in target file {filepath}: {e}")
            return

        clean_tree = _CleanTransformer(target_names).visit(tree)
        ast.fix_missing_locations(clean_tree)

        if sys.version_info >= (3, 9):
            clean_code = ast.unparse(clean_tree)
        else:
            import astor
            clean_code = astor.to_source(clean_tree)

        output = f"%%load_clean\n{import_line}\n\n{clean_code}"
        get_ipython().set_next_input(output, replace=True)

    @staticmethod
    def _parse_import_line(import_line):
        """Parse a literal `import x.y.z` or `from x.y.z import a, b as c` line."""
        try:
            node = ast.parse(import_line, mode="exec").body[0]
        except (SyntaxError, IndexError):
            return None, set()

        if isinstance(node, ast.Import) and node.names:
            return node.names[0].name, set()
        if isinstance(node, ast.ImportFrom) and node.module:
            # use original names (alias.name), not local aliases (alias.asname),
            # since we're matching definitions in the source file
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