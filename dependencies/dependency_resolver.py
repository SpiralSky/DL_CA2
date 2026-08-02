"""
Shared dependency resolution logic for %%load_clean magic and build script.

Uses scope-aware static analysis (like pyflakes) to correctly identify
unbound name references, avoiding false positives from loop variables,
comprehension targets, local assignments, etc.
"""
import ast
import importlib.util
import os
from typing import Optional, Set, Dict, List, Tuple, NamedTuple


class ImportInfo(NamedTuple):
    name: str
    module: Optional[str]
    is_external: bool
    import_line: str


class ResolutionResult(NamedTuple):
    needed: Set[str]
    external_refs: Set[str]
    internal_refs: Set[str]
    module_imports: List[ImportInfo]


# Builtins that don't need importing
BUILTINS = {
    'None', 'True', 'False', 'bool', 'int', 'float', 'str', 'list', 'dict',
    'set', 'tuple', 'object', 'type', 'len', 'range', 'enumerate', 'zip',
    'map', 'filter', 'sum', 'min', 'max', 'abs', 'round', 'sorted',
    'reversed', 'any', 'all', 'hasattr', 'getattr', 'setattr', 'isinstance',
    'issubclass', 'callable', 'print', 'input', 'open', 'help', 'dir',
    'vars', 'locals', 'globals', 'exec', 'eval', 'compile', 'chr', 'ord',
    'hex', 'oct', 'bin', 'pow', 'divmod', 'complex', 'bytes', 'bytearray',
    'memoryview', 'property', 'classmethod', 'staticmethod', 'super',
    'Exception', 'BaseException', 'Warning', 'TypeError', 'ValueError',
    'KeyError', 'IndexError', 'AttributeError', 'NameError', 'RuntimeError',
    'ImportError', 'ModuleNotFoundError', 'StopIteration', 'GeneratorExit',
    'SystemExit', 'KeyboardInterrupt', 'ArithmeticError', 'LookupError',
    'AssertionError', 'BufferError', 'EOFError', 'FloatingPointError',
    'OSError', 'IOError', 'EnvironmentError', 'BlockingIOError',
    'ChildProcessError', 'ConnectionError', 'BrokenPipeError',
    'ConnectionAbortedError', 'ConnectionRefusedError',
    'ConnectionResetError', 'FileExistsError', 'FileNotFoundError',
    'InterruptedError', 'IsADirectoryError', 'NotADirectoryError',
    'PermissionError', 'ProcessLookupError', 'TimeoutError',
    'ReferenceError', 'SyntaxError', 'IndentationError', 'TabError',
    'SystemError', 'UnicodeError', 'UnicodeDecodeError',
    'UnicodeEncodeError', 'UnicodeTranslateError', 'Warning',
    'UserWarning', 'DeprecationWarning', 'PendingDeprecationWarning',
    'SyntaxWarning', 'RuntimeWarning', 'FutureWarning', 'ImportWarning',
    'UnicodeWarning', 'BytesWarning', 'ResourceWarning',
    'Path', 'PathLike',
    'Optional', 'Union', 'List', 'Dict', 'Set', 'Tuple', 'Callable',
    'Any', 'Iterator', 'Iterable', 'Generator', 'NamedTuple',
    'Protocol', 'runtime_checkable', 'Final', 'Literal', 'TypedDict',
    'ClassVar', 'Annotated', 'TypeVar', 'Generic', 'Type',
    'Self', 'Never', 'NoReturn', 'Required', 'NotRequired', 'Unpack',
    'Concatenate', 'ParamSpec', 'ParamSpecArgs', 'ParamSpecKwargs',
    'TypeAlias', 'TypeGuard', 'TYPE_CHECKING', 'reveal_type',
    'cast', 'overload', 'final', 'dataclass', 'field',
    'ABC', 'abstractmethod', 'ABCMeta',
    # Additional builtins
    '__name__', '__file__', '__doc__', '__package__', '__spec__',
    '__annotations__', '__builtins__', '__cached__', '__loader__',
    'Ellipsis', 'NotImplemented', 'quit', 'exit', 'copyright', 'credits',
    'license', 'next', 'iter', 'slice', 'format', 'repr', 'ascii',
    'hash', 'id', 'oct', 'hex', 'bin', 'bool', 'bytearray', 'bytes',
    'classmethod', 'complex', 'delattr', 'dict', 'dir', 'divmod',
    'enumerate', 'filter', 'float', 'frozenset', 'getattr', 'globals',
    'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
    'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow',
    'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
    'tuple', 'type', 'vars', 'zip', '__import__',
}


class ScopeAnalyzer:
    """
    Scope-aware analyzer that tracks bindings per scope.
    Similar to how pyflakes works.
    """

    def __init__(self, tree: ast.AST):
        self.tree = tree
        self.scopes: List[Set[str]] = [set()]  # Global scope
        self.unbound_refs: Set[str] = set()
        self.module_imports: List[ImportInfo] = []

    def analyze(self) -> Tuple[Set[str], List[ImportInfo]]:
        """Returns (unbound_refs, module_imports)."""
        for node in self.tree.body:
            self._visit(node, is_module_level=True)
        return self.unbound_refs, self.module_imports

    def _visit(self, node: ast.AST, is_module_level: bool = False):
        """Visit a node, tracking scope."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_function(node)
        elif isinstance(node, ast.ClassDef):
            self._visit_class(node)
        elif isinstance(node, ast.Import):
            self._visit_import(node)
        elif isinstance(node, ast.ImportFrom):
            self._visit_import_from(node)
        elif isinstance(node, ast.Assign):
            self._visit_assign(node, is_module_level)
        elif isinstance(node, ast.AnnAssign):
            self._visit_ann_assign(node, is_module_level)
        elif isinstance(node, ast.AugAssign):
            self._visit_aug_assign(node)
        elif isinstance(node, ast.For):
            self._visit_for(node)
        elif isinstance(node, ast.While):
            self._visit_while(node)
        elif isinstance(node, ast.If):
            self._visit_if(node)
        elif isinstance(node, ast.With):
            self._visit_with(node)
        elif isinstance(node, ast.Try):
            self._visit_try(node)
        elif isinstance(node, ast.ExceptHandler):
            self._visit_except_handler(node)
        elif isinstance(node, ast.ListComp):
            self._visit_listcomp(node)
        elif isinstance(node, ast.SetComp):
            self._visit_setcomp(node)
        elif isinstance(node, ast.GeneratorExp):
            self._visit_generatorexp(node)
        elif isinstance(node, ast.DictComp):
            self._visit_dictcomp(node)
        elif isinstance(node, ast.Lambda):
            self._visit_lambda(node)
        elif isinstance(node, ast.NamedExpr):
            self._visit_named_expr(node)
        elif isinstance(node, ast.Expr):
            self._visit_expr(node.value)
        elif isinstance(node, ast.Return):
            if node.value:
                self._visit_expr(node.value)
        elif isinstance(node, ast.Raise):
            if node.exc:
                self._visit_expr(node.exc)
            if node.cause:
                self._visit_expr(node.cause)
        elif isinstance(node, ast.Assert):
            self._visit_expr(node.test)
            if node.msg:
                self._visit_expr(node.msg)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                self._visit_expr(target)
        elif isinstance(node, ast.Global):
            # global x -> binds x in global scope
            for name in node.names:
                self.scopes[0].add(name)
        elif isinstance(node, ast.Nonlocal):
            pass  # nonlocal doesn't create new binding, just references outer
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Break):
            pass
        elif isinstance(node, ast.Continue):
            pass
        elif isinstance(node, ast.Match):
            self._visit_match(node)
        else:
            # Generic fallback: visit all child nodes
            for child in ast.iter_child_nodes(node):
                self._visit(child, is_module_level)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        # Function name is defined in current scope
        if len(self.scopes) == 1:  # Module level
            self.scopes[-1].add(node.name)

        # Visit decorators in current scope
        for dec in node.decorator_list:
            self._visit_expr(dec)

        # Visit default values and annotations in current scope
        self._visit_arguments(node.args)
        if node.returns:
            self._visit_expr(node.returns)

        # New scope for function body
        self.scopes.append(set())

        # Add parameters to function scope
        self._add_args_to_scope(node.args)

        # Visit body in new scope
        for stmt in node.body:
            self._visit(stmt)

        self.scopes.pop()

    def _visit_class(self, node: ast.ClassDef):
        # Class name is defined in current scope
        if len(self.scopes) == 1:
            self.scopes[-1].add(node.name)

        # Visit decorators and bases in current scope
        for dec in node.decorator_list:
            self._visit_expr(dec)
        for base in node.bases:
            self._visit_expr(base)
        for keyword in node.keywords:
            self._visit_expr(keyword.value)

        # New scope for class body
        self.scopes.append(set())

        for stmt in node.body:
            self._visit(stmt)

        self.scopes.pop()

    def _visit_import(self, node: ast.Import):
        for alias in node.names:
            local_name = alias.asname or alias.name.split('.')[0]
            self.scopes[-1].add(local_name)
            if len(self.scopes) == 1:  # Module level
                is_ext = self._is_external_module(alias.name)
                self.module_imports.append(ImportInfo(
                    name=local_name,
                    module=alias.name,
                    is_external=is_ext,
                    import_line=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                ))

    def _visit_import_from(self, node: ast.ImportFrom):
        module = node.module or ""
        is_ext = self._is_external_module(module)
        for alias in node.names:
            if alias.name == '*':
                continue
            local_name = alias.asname or alias.name
            self.scopes[-1].add(local_name)
            if len(self.scopes) == 1:
                self.module_imports.append(ImportInfo(
                    name=local_name,
                    module=module,
                    is_external=is_ext,
                    import_line=f"from {module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                ))

    def _visit_assign(self, node: ast.Assign, is_module_level: bool = False):
        # Visit value first (expressions are evaluated before assignment)
        self._visit_expr(node.value)

        # Add targets to scope
        for target in node.targets:
            self._add_binding(target)

    def _visit_ann_assign(self, node: ast.AnnAssign, is_module_level: bool = False):
        # Visit annotation
        self._visit_expr(node.annotation)

        # Visit value if exists
        if node.value:
            self._visit_expr(node.value)

        # Add target to scope
        self._add_binding(node.target)

    def _visit_aug_assign(self, node: ast.AugAssign):
        # Visit value first
        self._visit_expr(node.value)
        # Visit target (it's both Load and Store, but we add to scope)
        self._visit_expr(node.target)
        self._add_binding(node.target)

    def _visit_for(self, node: ast.For):
        # Visit iter first
        self._visit_expr(node.iter)

        # Add loop variables to scope
        self._add_binding(node.target)

        # Visit body
        for stmt in node.body:
            self._visit(stmt)
        for stmt in node.orelse:
            self._visit(stmt)

    def _visit_while(self, node: ast.While):
        self._visit_expr(node.test)
        for stmt in node.body:
            self._visit(stmt)
        for stmt in node.orelse:
            self._visit(stmt)

    def _visit_if(self, node: ast.If):
        self._visit_expr(node.test)
        for stmt in node.body:
            self._visit(stmt)
        for stmt in node.orelse:
            self._visit(stmt)

    def _visit_with(self, node: ast.With):
        for item in node.items:
            self._visit_expr(item.context_expr)
            if item.optional_vars:
                self._add_binding(item.optional_vars)
        for stmt in node.body:
            self._visit(stmt)

    def _visit_try(self, node: ast.Try):
        for stmt in node.body:
            self._visit(stmt)
        for handler in node.handlers:
            self._visit(handler)
        for stmt in node.orelse:
            self._visit(stmt)
        for stmt in node.finalbody:
            self._visit(stmt)

    def _visit_except_handler(self, node: ast.ExceptHandler):
        if node.type:
            self._visit_expr(node.type)
        if node.name:
            self.scopes[-1].add(node.name)
        for stmt in node.body:
            self._visit(stmt)

    def _visit_match(self, node: ast.Match):
        self._visit_expr(node.subject)
        for case in node.cases:
            self._visit_match_case(case)

    def _visit_match_case(self, node: ast.match_case):
        self._visit_pattern(node.pattern)
        if node.guard:
            self._visit_expr(node.guard)
        for stmt in node.body:
            self._visit(stmt)

    def _visit_pattern(self, node: ast.AST):
        """Visit match patterns, adding bindings."""
        if isinstance(node, ast.MatchValue):
            self._visit_expr(node.value)
        elif isinstance(node, ast.MatchSequence):
            for p in node.patterns:
                self._visit_pattern(p)
        elif isinstance(node, ast.MatchMapping):
            for key in node.keys:
                self._visit_expr(key)
            for p in node.patterns:
                self._visit_pattern(p)
            if node.rest:
                self.scopes[-1].add(node.rest)
        elif isinstance(node, ast.MatchClass):
            self._visit_expr(node.cls)
            for p in node.patterns:
                self._visit_pattern(p)
            for p in node.kwd_patterns:
                self._visit_pattern(p)
        elif isinstance(node, ast.MatchStar):
            if node.name:
                self.scopes[-1].add(node.name)
        elif isinstance(node, ast.MatchAs):
            if node.pattern:
                self._visit_pattern(node.pattern)
            if node.name:
                self.scopes[-1].add(node.name)
        elif isinstance(node, ast.MatchOr):
            for p in node.patterns:
                self._visit_pattern(p)

    def _visit_listcomp(self, node: ast.ListComp):
        self._visit_comprehension(node, node.elt)

    def _visit_setcomp(self, node: ast.SetComp):
        self._visit_comprehension(node, node.elt)

    def _visit_generatorexp(self, node: ast.GeneratorExp):
        self._visit_comprehension(node, node.elt)

    def _visit_dictcomp(self, node: ast.DictComp):
        self.scopes.append(set())
        for gen in node.generators:
            self._visit_expr(gen.iter)
            self._add_binding(gen.target)
            for if_clause in gen.ifs:
                self._visit_expr(if_clause)
        self._visit_expr(node.key)
        self._visit_expr(node.value)
        self.scopes.pop()

    def _visit_comprehension(self, node, elt_node: ast.AST):
        self.scopes.append(set())
        for gen in node.generators:
            self._visit_expr(gen.iter)
            self._add_binding(gen.target)
            for if_clause in gen.ifs:
                self._visit_expr(if_clause)
        self._visit_expr(elt_node)
        self.scopes.pop()

    def _visit_lambda(self, node: ast.Lambda):
        # Visit default values in current scope
        self._visit_arguments(node.args)

        self.scopes.append(set())
        self._add_args_to_scope(node.args)
        self._visit_expr(node.body)
        self.scopes.pop()

    def _visit_named_expr(self, node: ast.NamedExpr):
        self._visit_expr(node.value)
        self._add_binding(node.target)

    def _visit_expr(self, node: ast.AST):
        """Visit an expression, collecting unbound references."""
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self._check_ref(node.id)
        elif isinstance(node, ast.Attribute):
            self._visit_expr(node.value)
        elif isinstance(node, ast.Call):
            self._visit_expr(node.func)
            for arg in node.args:
                self._visit_expr(arg)
            for keyword in node.keywords:
                self._visit_expr(keyword.value)
        elif isinstance(node, ast.BinOp):
            self._visit_expr(node.left)
            self._visit_expr(node.right)
        elif isinstance(node, ast.UnaryOp):
            self._visit_expr(node.operand)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self._visit_expr(value)
        elif isinstance(node, ast.Compare):
            self._visit_expr(node.left)
            for comparator in node.comparators:
                self._visit_expr(comparator)
        elif isinstance(node, ast.IfExp):
            self._visit_expr(node.test)
            self._visit_expr(node.body)
            self._visit_expr(node.orelse)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                self._visit_expr(elt)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key:
                    self._visit_expr(key)
            for value in node.values:
                self._visit_expr(value)
        elif isinstance(node, ast.Subscript):
            self._visit_expr(node.value)
            self._visit_expr(node.slice)
        elif isinstance(node, ast.Starred):
            self._visit_expr(node.value)
        elif isinstance(node, ast.Slice):
            if node.lower:
                self._visit_expr(node.lower)
            if node.upper:
                self._visit_expr(node.upper)
            if node.step:
                self._visit_expr(node.step)
        elif isinstance(node, ast.FormattedValue):
            self._visit_expr(node.value)
            if node.format_spec:
                self._visit_expr(node.format_spec)
        elif isinstance(node, ast.JoinedStr):
            for value in node.values:
                self._visit_expr(value)
        elif isinstance(node, ast.NamedExpr):
            self._visit_expr(node.value)
            self._add_binding(node.target)
        elif isinstance(node, ast.Lambda):
            self._visit_lambda(node)
        elif isinstance(node, ast.ListComp):
            self._visit_listcomp(node)
        elif isinstance(node, ast.SetComp):
            self._visit_setcomp(node)
        elif isinstance(node, ast.GeneratorExp):
            self._visit_generatorexp(node)
        elif isinstance(node, ast.DictComp):
            self._visit_dictcomp(node)
        elif isinstance(node, ast.Await):
            self._visit_expr(node.value)
        elif isinstance(node, ast.Yield):
            if node.value:
                self._visit_expr(node.value)
        elif isinstance(node, ast.YieldFrom):
            self._visit_expr(node.value)
        # ast.Constant, ast.Expr with string docstrings, etc. don't contain refs

    def _check_ref(self, name: str):
        """Check if a name reference is bound in any enclosing scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return
        self.unbound_refs.add(name)

    def _add_binding(self, node: ast.AST):
        """Add a binding to the current scope."""
        if isinstance(node, ast.Name):
            self.scopes[-1].add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._add_binding(elt)
        elif isinstance(node, ast.Starred):
            self._add_binding(node.value)
        elif isinstance(node, ast.Attribute):
            # Can't bind to attribute (e.g. `obj.attr = 1`)
            pass
        elif isinstance(node, ast.Subscript):
            # Can't bind to subscript
            pass

    def _add_args_to_scope(self, args: ast.arguments):
        """Add function parameters to current scope."""
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            self.scopes[-1].add(arg.arg)
        if args.vararg:
            self.scopes[-1].add(args.vararg.arg)
        if args.kwarg:
            self.scopes[-1].add(args.kwarg.arg)

    def _visit_arguments(self, args: ast.arguments):
        """Visit default values and annotations in arguments."""
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if arg.annotation:
                self._visit_expr(arg.annotation)
        if args.vararg and args.vararg.annotation:
            self._visit_expr(args.vararg.annotation)
        if args.kwarg and args.kwarg.annotation:
            self._visit_expr(args.kwarg.annotation)
        for default in args.defaults + args.kw_defaults:
            if default:
                self._visit_expr(default)

    @staticmethod
    def _is_external_module(module_path: str) -> bool:
        if not module_path:
            return True
        local_prefixes = ('src.', 'scripts.', 'notebooks.', 'project.', 'app.')
        if module_path.startswith(local_prefixes):
            return False
        try:
            spec = importlib.util.find_spec(module_path.split('.')[0])
            if spec is None:
                return True
            if spec.origin:
                origin = str(spec.origin)
                if 'site-packages' in origin or 'lib/python' in origin:
                    return True
                cwd = os.getcwd()
                if origin.startswith(cwd):
                    return False
            return True
        except (ModuleNotFoundError, ValueError, ImportError):
            return True


class DependencyResolver:
    """
    Given a module source and optional target names, resolves:
    1. Which top-level definitions must be included (transitive closure)
    2. Which external names are referenced but not defined/imported in-module
    3. Categorizes missing refs as external (stdlib/pypi) vs internal (src.*)
    """

    def __init__(self, source: str, target_names: Optional[Set[str]] = None):
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.tree = ast.parse(source)
        # Treat {"*"} as empty (wildcard = include everything)
        self.target_names = target_names or set()
        if self.target_names == {"*"}:
            self.target_names = set()
        self._name_to_node: Dict[str, ast.AST] = {}
        self._build_index()

    def _build_index(self):
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._name_to_node[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._name_to_node[target.id] = node

    def resolve(self) -> ResolutionResult:
        # Analyze the entire module for unbound refs and imports
        analyzer = ScopeAnalyzer(self.tree)
        all_unbound, all_imports = analyzer.analyze()

        # If no specific targets OR wildcard import (*), include everything
        if not self.target_names or '*' in self.target_names:
            all_names = set(self._name_to_node.keys())
            external, internal = self._categorize_missing(all_unbound, all_imports)
            return ResolutionResult(all_names, external, internal, all_imports)

        # Resolve transitive closure for specific targets
        needed = set(self.target_names)
        queue = list(self.target_names)
        seen = set()
        target_unbound: Set[str] = set()

        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)

            node = self._name_to_node.get(name)
            if node is None:
                continue

            # Analyze just this node's unbound refs
            node_analyzer = ScopeAnalyzer(ast.Module(body=[node], type_ignores=[]))
            node_unbound, _ = node_analyzer.analyze()
            target_unbound.update(node_unbound)

            for ref in node_unbound:
                if ref in self._name_to_node and ref not in needed:
                    needed.add(ref)
                    queue.append(ref)

        external, internal = self._categorize_missing(target_unbound, all_imports)
        return ResolutionResult(needed, external, internal, all_imports)

    def _categorize_missing(self, unbound: Set[str], imports: List[ImportInfo]) -> Tuple[Set[str], Set[str]]:
        imported_names = {imp.name for imp in imports}
        defined_names = set(self._name_to_node.keys())

        missing = unbound - defined_names - imported_names - BUILTINS

        external: Set[str] = set()
        internal: Set[str] = set()

        for ref in missing:
            external.add(ref)

        return external, internal

    def slice_source(self, names: Set[str]) -> str:
        kept: List[ast.AST] = []
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in names:
                    kept.append(node)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        kept.append(node)
                        break

        if not kept:
            return ""

        segments = []
        for node in kept:
            start = self._effective_start_line(node)
            end = node.end_lineno
            raw = "".join(self.lines[start - 1:end])
            if not raw.endswith("\n"):
                raw += "\n"
            segments.append(raw)

        return "".join(segments).strip("\n") + "\n"

    @staticmethod
    def _effective_start_line(node: ast.AST) -> int:
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            return decorators[0].lineno
        return node.lineno


def resolve_module_path(module_path: str) -> Optional[str]:
    filepath = None
    try:
        spec = importlib.util.find_spec(module_path)
        if spec and spec.origin:
            filepath = spec.origin
    except (ModuleNotFoundError, ValueError):
        pass
    if not filepath:
        fallback = module_path.replace(".", "/") + ".py"
        if os.path.exists(fallback):
            filepath = fallback
    if filepath and os.path.exists(filepath):
        return filepath
    return None


def parse_import_line(import_line: str) -> Tuple[Optional[str], Set[str]]:
    try:
        tree = ast.parse(import_line, mode="exec")
        node = tree.body[0]
    except (SyntaxError, IndexError):
        return None, set()

    if isinstance(node, ast.Import) and node.names:
        return node.names[0].name, set()
    if isinstance(node, ast.ImportFrom) and node.module:
        targets = {alias.name for alias in node.names}
        return node.module, targets
    return None, set()


def normalize_import_line(import_line: str) -> str:
    module_path, targets = parse_import_line(import_line)
    if module_path is None:
        return import_line
    if not targets:
        return f"from {module_path} import *"
    return import_line


def check_dependencies(
    external_refs: Set[str],
    internal_refs: Set[str],
    user_ns: dict,
    module_path: str,
    module_imports: List[ImportInfo],
) -> List[str]:
    warnings = []

    for ref in sorted(external_refs):
        if ref not in user_ns:
            suggestion = ""
            for imp in module_imports:
                if imp.name == ref and imp.is_external:
                    suggestion = f" (try: {imp.import_line})"
                    break
            warnings.append(
                f"[DEP-WARN] '{ref}' (needed by {module_path}) not found in notebook namespace.{suggestion}"
            )

    for ref in sorted(internal_refs):
        if ref not in user_ns:
            warnings.append(
                f"[DEP-ERROR] '{ref}' (needed by {module_path}) not found in notebook namespace. "
                f"This is an internal module reference — did you forget to %%load_clean it first?"
            )

    return warnings
