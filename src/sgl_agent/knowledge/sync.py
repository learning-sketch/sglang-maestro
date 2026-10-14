"""Build a capability snapshot from an SGLang source checkout.

Everything here is static analysis (``ast``): SGLang itself is never imported, so the
snapshot can be regenerated on a machine without GPUs or SGLang's dependencies.
"""

from __future__ import annotations

import ast
import gc
import json
import re
import subprocess
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA_VERSION = 1

_UNRESOLVED = object()

_MOE_CONFIG_NAME_RE = re.compile(
    r"^E=(?P<experts>\d+),N=(?P<intermediate>\d+),device_name=(?P<device>[^,]+?)"
    r"(?:,dtype=(?P<dtype>[^,]+?))?"
    r"(?:,block_shape=\[(?P<block_n>\d+), (?P<block_k>\d+)\])?"
    r"(?P<per_channel>,per_channel_quant=True)?"
    r"(?P<down>_down)?\.json$"
)


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return _UNRESOLVED
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _UNRESOLVED


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    return ""


def _resolve_collection(node: ast.AST, constants: dict[str, list]) -> Any:
    """Resolve list/tuple/set literals, ``frozenset({...})`` calls and references to known constants."""
    if isinstance(node, ast.Name) and node.id in constants:
        return list(constants[node.id])
    if (
        isinstance(node, ast.Call)
        and _dotted_name(node.func) in {"frozenset", "set", "list", "tuple", "sorted"}
        and len(node.args) == 1
    ):
        inner = _resolve_collection(node.args[0], constants)
        if isinstance(inner, (list, tuple, set, frozenset)):
            return sorted(inner) if not isinstance(inner, list) else inner
        return _UNRESOLVED
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve_collection(node.left, constants)
        right = _resolve_collection(node.right, constants)
        if isinstance(left, list) and isinstance(right, list):
            return left + right
        return _UNRESOLVED
    value = _literal(node)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def extract_choice_constants(choices_path: Path) -> dict[str, list]:
    tree = ast.parse(choices_path.read_text(encoding="utf-8"))
    constants: dict[str, list] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        resolved = _resolve_collection(value, constants)
        if isinstance(resolved, list):
            constants[target.id] = resolved
    return constants


def _literal_choices_from_type(type_node: ast.AST) -> list | None:
    """Return the values of ``Literal[...]`` (also inside ``Optional[...]``) or None."""
    if isinstance(type_node, ast.Subscript):
        outer = _dotted_name(type_node.value).split(".")[-1]
        if outer == "Literal":
            elements = (
                type_node.slice.elts
                if isinstance(type_node.slice, ast.Tuple)
                else [type_node.slice]
            )
            values = [_literal(element) for element in elements]
            if all(value is not _UNRESOLVED for value in values):
                return values
            return None
        if outer in {"Optional", "Union"}:
            elements = (
                type_node.slice.elts
                if isinstance(type_node.slice, ast.Tuple)
                else [type_node.slice]
            )
            for element in elements:
                found = _literal_choices_from_type(element)
                if found is not None:
                    return found
    return None


def _first_sentence(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def extract_server_flags(
    fields_dir: Path, constants: dict[str, list], repo_root: Path
) -> dict[str, dict[str, Any]]:
    """Map every CLI flag declared through ``A[T, Arg(...)]`` fields to its metadata."""
    flags: dict[str, dict[str, Any]] = {}
    for path in sorted(fields_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for statement in class_node.body:
                if not isinstance(statement, ast.AnnAssign) or not isinstance(
                    statement.target, ast.Name
                ):
                    continue
                annotation = statement.annotation
                if not (
                    isinstance(annotation, ast.Subscript)
                    and _dotted_name(annotation.value).split(".")[-1] in {"A", "Annotated"}
                ):
                    continue
                elements = (
                    annotation.slice.elts
                    if isinstance(annotation.slice, ast.Tuple)
                    else [annotation.slice]
                )
                type_node, metadata_nodes = elements[0], elements[1:]
                help_text = ""
                choices: Any = None
                aliases: list[str] = []
                cli_name: str | None = None
                no_cli = False
                boolean_optional = False
                for metadata in metadata_nodes:
                    if isinstance(metadata, ast.Constant) and isinstance(metadata.value, str):
                        help_text = metadata.value
                    elif isinstance(metadata, ast.Call) and _dotted_name(metadata.func).endswith(
                        "Arg"
                    ):
                        for keyword in metadata.keywords:
                            value = _literal(keyword.value)
                            if keyword.arg == "help" and isinstance(value, str):
                                help_text = value
                            elif keyword.arg == "choices":
                                resolved = _resolve_collection(keyword.value, constants)
                                choices = resolved if isinstance(resolved, list) else None
                            elif keyword.arg == "aliases" and isinstance(value, list):
                                aliases = [str(alias) for alias in value]
                            elif keyword.arg == "cli_name" and isinstance(value, str):
                                cli_name = value
                            elif keyword.arg == "no_cli" and value is True:
                                no_cli = True
                            elif keyword.arg == "action" and "BooleanOptionalAction" in ast.unparse(
                                keyword.value
                            ):
                                boolean_optional = True
                if no_cli:
                    continue
                if choices is None:
                    choices = _literal_choices_from_type(type_node)
                field_name = statement.target.id
                flag = cli_name or "--" + field_name.replace("_", "-")
                default = _literal(statement.value) if statement.value is not None else None
                type_text = ast.unparse(type_node)
                if boolean_optional:
                    aliases = aliases + ["--no-" + flag.removeprefix("--")]
                flags[flag] = {
                    "field": field_name,
                    "type": type_text,
                    "is_bool": type_text in {"bool", "Optional[bool]"},
                    "default": None if default is _UNRESOLVED else default,
                    "choices": choices,
                    "aliases": aliases,
                    "help": _first_sentence(help_text),
                    "source": f"{path.relative_to(repo_root).as_posix()}:{statement.lineno}",
                }
    return flags


def extract_manual_flags(server_args_path: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    """Flags registered by explicit ``parser.add_argument("--x", ...)`` calls in server_args.py."""
    tree = ast.parse(server_args_path.read_text(encoding="utf-8"))
    flags: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        names = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        ]
        if not names:
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        help_value = _literal(keywords.get("help"))
        help_text = help_value if isinstance(help_value, str) else ""
        action_text = ast.unparse(keywords["action"]) if "action" in keywords else ""
        deprecated = "Deprecated" in action_text or help_text.lstrip("( ").startswith("Deprecated")
        replacement = _literal(keywords.get("new_flag"))
        flags[names[0]] = {
            "field": None,
            "type": "manual",
            "is_bool": "store_true" in action_text or "StoreTrue" in action_text,
            "default": None,
            "choices": None,
            "aliases": names[1:],
            "help": _first_sentence(help_text),
            "deprecated": deprecated,
            "replacement": replacement if isinstance(replacement, str) else None,
            "source": f"{server_args_path.relative_to(repo_root).as_posix()}:{node.lineno}",
        }
    return flags


def _entry_class_names(module: ast.Module) -> list[str]:
    """Resolve ``EntryClass = X`` / ``EntryClass = [X, Y]`` to class ``__name__`` values."""
    aliases: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Name)
            and node.targets[0].id != "EntryClass"
        ):
            aliases[node.targets[0].id] = node.value.id

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in aliases and name not in seen and aliases[name] != name:
            seen.add(name)
            name = aliases[name]
        return name

    def names_in(node: ast.AST) -> list[str]:
        if isinstance(node, ast.Name):
            return [resolve(node.id)]
        if isinstance(node, (ast.List, ast.Tuple)):
            collected: list[str] = []
            for element in node.elts:
                collected.extend(names_in(element))
            return collected
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return names_in(node.left) + names_in(node.right)
        return []

    found: list[str] = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "EntryClass" for t in node.targets
        ):
            found.extend(names_in(node.value))
    return found


def extract_model_architectures(models_dir: Path) -> dict[str, str]:
    architectures: dict[str, str] = {}
    for path in sorted(models_dir.glob("*.py")):
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for class_name in _entry_class_names(module):
            architectures.setdefault(class_name, path.stem)
    return dict(sorted(architectures.items()))


def extract_moe_tuned_configs(configs_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for version_dir in sorted(configs_root.glob("triton_*")):
        if not version_dir.is_dir():
            continue
        triton_version = version_dir.name.removeprefix("triton_").replace("_", ".")
        for config_path in sorted(version_dir.glob("*.json")):
            match = _MOE_CONFIG_NAME_RE.match(config_path.name)
            if not match:
                continue
            entries.append(
                {
                    "triton": triton_version,
                    "E": int(match["experts"]),
                    "N": int(match["intermediate"]),
                    "device": match["device"],
                    "dtype": match["dtype"],
                    "block_shape": [int(match["block_n"]), int(match["block_k"])]
                    if match["block_n"]
                    else None,
                    "per_channel_quant": bool(match["per_channel"]),
                    "down": bool(match["down"]),
                }
            )
    return entries


def _frozenset_strings(path: Path, variable: str) -> list[str]:
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == variable
        ):
            resolved = _resolve_collection(node.value, {})
            if isinstance(resolved, list):
                return [str(item) for item in resolved]
    return []


_MODELS_PACKAGE = "sglang.srt.models"


def _imported_modules(module: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 1:
                imported.add(f"{_MODELS_PACKAGE}.{node.module}")
            elif node.level == 0:
                imported.add(node.module)
    return imported


def _architectures_in_files_matching(models_dir: Path, pattern: re.Pattern[str]) -> list[str]:
    """``EntryClass`` names of model files whose source matches ``pattern``.

    Top-level model files also count when they import a helper module under a model subpackage
    (e.g. ``models/dots3_common/modeling.py``) that matches.
    """
    matching_helpers = {
        f"{_MODELS_PACKAGE}." + ".".join(path.relative_to(models_dir).with_suffix("").parts)
        for path in models_dir.rglob("*.py")
        if path.parent != models_dir and pattern.search(path.read_text(encoding="utf-8"))
    }
    found: set[str] = set()
    for path in sorted(models_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        direct = pattern.search(text) is not None
        if not direct and not matching_helpers:
            continue
        try:
            module = ast.parse(text)
        except SyntaxError:
            continue
        if direct or _imported_modules(module) & matching_helpers:
            found.update(_entry_class_names(module))
    return sorted(found)


def extract_tbo_layer_classes(strategy_path: Path) -> list[str]:
    """Decoder-layer class names ``OperationsStrategy.init_new_tbo`` has a strategy for."""
    if not strategy_path.exists():
        return []
    text = strategy_path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'layer_name == "(\w+)"', text)))


def _instantiation_pattern(class_names: list[str]) -> re.Pattern[str] | None:
    """Matches ``Name(`` calls; subclass declarations (``class X(Name)``) do not match."""
    if not class_names:
        return None
    return re.compile(r"\b(?:" + "|".join(map(re.escape, class_names)) + r")\(")


FUSED_OP_BASE = "sglang.kernels.fused_op.BaseFusedOp"
_RUNTIME_PACKAGES = ("sglang/srt", "sglang/kernels")
_PLATFORMS_PACKAGE = Path("sglang/srt/platforms")
_MAX_ALIAS_HOPS = 16

ParsedModule = tuple[Path, ast.Module]


def _module_qualname(path: Path, python_root: Path) -> str:
    parts = list(path.relative_to(python_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_source(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package_parts = module.split(".") if is_package else module.split(".")[:-1]
    if node.level > 1:
        package_parts = package_parts[: max(len(package_parts) - (node.level - 1), 0)]
    return ".".join(package_parts + ([node.module] if node.module else []))


def _module_level_statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    """Top-level statements, including those nested in ``if``/``try``/``with`` blocks."""
    collected: list[ast.stmt] = []
    for statement in statements:
        collected.append(statement)
        if isinstance(statement, ast.If):
            collected.extend(_module_level_statements(statement.body + statement.orelse))
        elif isinstance(statement, ast.Try):
            handlers = [line for handler in statement.handlers for line in handler.body]
            collected.extend(
                _module_level_statements(
                    statement.body + handlers + statement.orelse + statement.finalbody
                )
            )
        elif isinstance(statement, ast.With):
            collected.extend(_module_level_statements(statement.body))
    return collected


def _parse_quietly(path: Path) -> ast.Module | None:
    """Parse a source file, ignoring invalid-escape ``SyntaxWarning``s in third-party code."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


def _is_test_file(relative_parts: tuple[str, ...]) -> bool:
    return any(part in {"test", "tests"} for part in relative_parts) or relative_parts[
        -1
    ].startswith("test_")


@contextmanager
def _cyclic_gc_paused() -> Iterator[None]:
    """Thousands of live AST trees (which hold no reference cycles) make every full collection
    rescan millions of nodes; pausing the cyclic GC keeps parsing and extraction ~4x faster."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def parse_runtime_modules(python_root: Path) -> list[ParsedModule]:
    """``(path relative to python_root, tree)`` for every non-test module of the serving runtime."""
    modules: list[ParsedModule] = []
    with _cyclic_gc_paused():
        for package in _RUNTIME_PACKAGES:
            for path in sorted((python_root / package).rglob("*.py")):
                relative = path.relative_to(python_root)
                if _is_test_file(relative.parts):
                    continue
                tree = _parse_quietly(path)
                if tree is not None:
                    modules.append((relative, tree))
    return modules


def _only_raises_not_implemented(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
        return False
    raised = body[0].exc
    target = raised.func if isinstance(raised, ast.Call) else raised
    return _dotted_name(target).split(".")[-1] == "NotImplementedError"


def _forward_override(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """How an overriding ``forward()`` reaches kernels: via fused-op dispatch or hard-wired."""
    calls_dispatch = False
    direct_forwards: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver, attribute = node.func.value, node.func.attr
        if attribute == "forward" and (
            (isinstance(receiver, ast.Call) and _dotted_name(receiver.func) == "super")
            or (isinstance(receiver, ast.Name) and receiver.id != "self")
        ):
            calls_dispatch = True
        elif (
            attribute.startswith("forward_")
            and isinstance(receiver, ast.Name)
            and receiver.id == "self"
        ):
            direct_forwards.add(attribute)
    return {"calls_dispatch": calls_dispatch, "direct_forwards": sorted(direct_forwards)}


def _c3_merge(sequences: list[list[str]]) -> list[str] | None:
    pending = [list(sequence) for sequence in sequences if sequence]
    merged: list[str] = []
    while pending:
        for sequence in pending:
            head = sequence[0]
            if not any(head in other[1:] for other in pending):
                break
        else:
            return None
        merged.append(head)
        pending = [
            sequence[1:] if sequence[0] == head else sequence
            for sequence in pending
            if sequence[1:] or sequence[0] != head
        ]
    return merged


def _linearize(
    name: str, classes: dict[str, dict[str, Any]], cache: dict[str, list[str]]
) -> list[str]:
    """C3 method resolution order over the scanned classes; unknown bases are treated as leaves."""
    if name in cache:
        return cache[name]
    cache[name] = [name]
    entry = classes.get(name)
    if entry is None or not entry["bases"]:
        return cache[name]
    base_orders = [_linearize(base, classes, cache) for base in entry["bases"]]
    merged = _c3_merge([*base_orders, list(entry["bases"])])
    if merged is None:
        merged = list(dict.fromkeys(item for order in base_orders for item in order))
    cache[name] = [name, *merged]
    return cache[name]


def _module_local_names(
    module: str, is_package: bool, statements: list[ast.stmt]
) -> dict[str, str]:
    """Module-level names mapped to the qualified name they refer to (classes, imports, aliases)."""
    local_names: dict[str, str] = {}
    for statement in statements:
        if isinstance(statement, ast.ClassDef):
            local_names[statement.name] = f"{module}.{statement.name}"
    for statement in statements:
        if isinstance(statement, ast.ImportFrom):
            source = _import_source(module, is_package, statement)
            for alias in statement.names:
                if alias.name != "*":
                    local_names[alias.asname or alias.name] = f"{source}.{alias.name}"
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.asname:
                    local_names[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    local_names.setdefault(head, head)
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Name)
        ):
            value_name = statement.value.id
            local_names[statement.targets[0].id] = local_names.get(
                value_name, f"{module}.{value_name}"
            )
    return local_names


def _class_record(
    class_node: ast.ClassDef, relative: Path, local_names: dict[str, str]
) -> dict[str, Any]:
    def resolve_base(node: ast.expr) -> str | None:
        dotted = _dotted_name(node)
        if not dotted:
            return None
        head, _, rest = dotted.partition(".")
        if head in local_names:
            return local_names[head] + (f".{rest}" if rest else "")
        return dotted

    forwards: set[str] = set()
    stubs: set[str] = set()
    forward_override: dict[str, Any] | None = None
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "forward":
                forward_override = _forward_override(item)
            elif item.name.startswith("forward_"):
                forwards.add(item.name)
                if _only_raises_not_implemented(item):
                    stubs.add(item.name)
        elif isinstance(item, ast.Assign):
            forwards.update(
                target.id
                for target in item.targets
                if isinstance(target, ast.Name) and target.id.startswith("forward_")
            )
    return {
        "file": (Path("python") / relative).as_posix(),
        "line": class_node.lineno,
        "bases": [base for base in map(resolve_base, class_node.bases) if base],
        "forwards": forwards,
        "stubs": stubs,
        "forward_override": forward_override,
    }


def extract_fused_op_classes(
    python_root: Path, modules: list[ParsedModule] | None = None
) -> dict[str, dict[str, Any]]:
    """Every transitive subclass of ``BaseFusedOp`` with the ``forward_*`` methods it can dispatch to.

    Base names are resolved through each module's imports (and re-exports), so same-named classes
    in different modules stay distinct. ``mro`` stops before ``BaseFusedOp`` because that is where
    ``BaseFusedOp._defined_method`` stops looking. ``forwards`` are the class's own methods,
    ``effective_forwards`` include inherited ones, and ``stub_forwards`` are effective methods whose
    resolving definition only raises ``NotImplementedError``. ``forward_override`` describes a
    ``forward()`` that replaces ``BaseFusedOp.forward``: whether it still calls into dispatch
    (``super().forward`` / ``Base.forward``) and which ``self.forward_*`` it calls directly; paths
    that do neither bypass OOT overrides. OOT dispatch matches ``register_oot_forward`` entries on
    the exact class, while ``forward_<key>`` methods are inherited along the MRO.
    """
    aliases: dict[str, str] = {}
    classes: dict[str, dict[str, Any]] = {}
    for relative, tree in modules if modules is not None else parse_runtime_modules(python_root):
        module = _module_qualname(python_root / relative, python_root)
        statements = _module_level_statements(tree.body)
        local_names = _module_local_names(module, relative.name == "__init__.py", statements)
        for name, target in local_names.items():
            own = f"{module}.{name}"
            if target != own:
                aliases.setdefault(own, target)
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                classes[f"{module}.{statement.name}"] = _class_record(
                    statement, relative, local_names
                )

    def canonical(name: str) -> str:
        for _ in range(_MAX_ALIAS_HOPS):
            if name in classes or name not in aliases:
                return name
            name = aliases[name]
        return name

    for entry in classes.values():
        entry["bases"] = [canonical(base) for base in entry["bases"]]
    fused = {FUSED_OP_BASE}
    changed = True
    while changed:
        changed = False
        for name, entry in classes.items():
            if name not in fused and any(base in fused for base in entry["bases"]):
                fused.add(name)
                changed = True

    linearization_cache: dict[str, list[str]] = {}
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(fused - {FUSED_OP_BASE}):
        entry = classes.get(name)
        if entry is None:
            continue
        order = _linearize(name, classes, linearization_cache)
        mro = order[: order.index(FUSED_OP_BASE)] if FUSED_OP_BASE in order else order
        resolved_is_stub: dict[str, bool] = {}
        forward_override: dict[str, Any] | None = None
        for klass in mro:
            klass_entry = classes.get(klass)
            if klass_entry is None:
                continue
            for method in klass_entry["forwards"]:
                resolved_is_stub.setdefault(method, method in klass_entry["stubs"])
            if forward_override is None and klass_entry["forward_override"] is not None:
                forward_override = {"defined_in": klass, **klass_entry["forward_override"]}
        result[name] = {
            "file": entry["file"],
            "line": entry["line"],
            "bases": entry["bases"],
            "mro": mro,
            "forwards": sorted(entry["forwards"]),
            "effective_forwards": sorted(resolved_is_stub),
            "stub_forwards": sorted(method for method, stub in resolved_is_stub.items() if stub),
            "forward_override": forward_override,
        }
    return result


_PLATFORM_HOOK_CLASSES = (("device_mixin.py", "DeviceMixin"), ("interface.py", "SRTPlatform"))
_DOC_STATUS_RE = re.compile(r"\[(Active|Planned)\]\s*")


def _body_without_docstring(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _hook_default(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, Any]:
    """``(kind, literal)``: raises / noop / returns (with the literal) / implemented."""
    if _only_raises_not_implemented(function):
        return "raises", None
    body = _body_without_docstring(function)
    if all(
        isinstance(statement, ast.Pass)
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is Ellipsis
        )
        for statement in body
    ):
        return "noop", None
    if len(body) == 1 and isinstance(body[0], ast.Return):
        value = _literal(body[0].value) if body[0].value is not None else None
        if value is not _UNRESOLVED:
            return "returns", value
    return "implemented", None


def _is_platform_expression(node: ast.AST) -> bool:
    """``current_platform`` or ``<package>.current_platform`` (e.g. ``platforms.current_platform``)."""
    if isinstance(node, ast.Name):
        return node.id == "current_platform"
    return isinstance(node, ast.Attribute) and node.attr == "current_platform"


def _is_identity_query(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(node, ast.Attribute) and node.attr == "_enum" for node in ast.walk(function)
    ) or any(_dotted_name(decorator).endswith("property") for decorator in function.decorator_list)


def extract_platform_hooks(
    python_root: Path, modules: list[ParsedModule] | None = None
) -> dict[str, Any]:
    """Platform plugin interface: hooks, class attributes and capability flags, with call sites.

    ``doc_status`` is the ``[Active]``/``[Planned]`` tag from the docstring; ``call_sites`` are the
    accesses outside ``srt/platforms`` (tests excluded) through ``current_platform``,
    ``platforms.current_platform`` or a local name bound only to one of them. A hook tagged Planned
    can still have call sites, and an Active one can have none: trust the call sites.
    """
    hooks: dict[str, dict[str, Any]] = {}
    attributes: dict[str, dict[str, Any]] = {}
    capabilities: dict[str, dict[str, Any]] = {}
    for file_name, class_name in _PLATFORM_HOOK_CLASSES:
        relative = _PLATFORMS_PACKAGE / file_name
        tree = _parse_quietly(python_root / relative)
        if tree is None:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name == "PlatformCapabilities":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        default = _literal(item.value)
                        capabilities[item.target.id] = {
                            "default": None if default is _UNRESOLVED else default,
                            "source": f"python/{relative.as_posix()}:{item.lineno}",
                            "readers": [],
                        }
            if node.name != class_name:
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") or _is_identity_query(item):
                        continue
                    docstring = (ast.get_docstring(item) or "").strip()
                    status = _DOC_STATUS_RE.search(docstring)
                    first_line = docstring.splitlines()[0] if docstring else ""
                    kind, returned = _hook_default(item)
                    record: dict[str, Any] = {
                        "defined_in": class_name,
                        "source": f"python/{relative.as_posix()}:{item.lineno}",
                        "doc_status": status.group(1) if status else None,
                        "default": kind,
                        "summary": _first_sentence(_DOC_STATUS_RE.sub("", first_line), 160),
                        "call_sites": [],
                    }
                    if kind == "returns":
                        record["default_return"] = returned
                    hooks[item.name] = record
                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    target = item.targets[0] if isinstance(item, ast.Assign) else item.target
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        attributes[target.id] = {
                            "defined_in": class_name,
                            "source": f"python/{relative.as_posix()}:{item.lineno}",
                            "readers": [],
                        }
    platforms_prefix = _PLATFORMS_PACKAGE.parts
    for relative, tree in modules if modules is not None else parse_runtime_modules(python_root):
        if relative.parts[: len(platforms_prefix)] == platforms_prefix:
            continue
        if "current_platform" not in (python_root / relative).read_text(encoding="utf-8"):
            continue
        bound_to_platform: set[str] = set()
        bound_elsewhere: set[str] = set()
        accesses: list[ast.Attribute] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                accesses.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        is_platform = node.value is not None and _is_platform_expression(node.value)
                        (bound_to_platform if is_platform else bound_elsewhere).add(target.id)
        aliases = bound_to_platform - bound_elsewhere

        def is_platform(receiver: ast.AST, alias_names: set[str] = aliases) -> bool:
            return _is_platform_expression(receiver) or (
                isinstance(receiver, ast.Name) and receiver.id in alias_names
            )

        for node in accesses:
            location = f"python/{relative.as_posix()}:{node.lineno}"
            receiver = node.value
            if is_platform(receiver):
                if node.attr in hooks:
                    hooks[node.attr]["call_sites"].append(location)
                elif node.attr in attributes:
                    attributes[node.attr]["readers"].append(location)
            elif (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "capabilities"
                and is_platform(receiver.value)
                and node.attr in capabilities
            ):
                capabilities[node.attr]["readers"].append(location)
    for table, key in ((hooks, "call_sites"), (attributes, "readers"), (capabilities, "readers")):
        for record in table.values():
            record[key] = sorted(set(record[key]), key=_location_sort_key)
    return {"hooks": hooks, "attributes": attributes, "capabilities": capabilities}


def _location_sort_key(location: str) -> tuple[str, int]:
    path, _, line = location.rpartition(":")
    return (path, int(line)) if line.isdigit() else (location, 0)


def _enum_member_values(path: Path, class_name: str) -> dict[str, str]:
    tree = _parse_quietly(path)
    if tree is None:
        return {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: dict[str, str] = {}
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    values[statement.targets[0].id] = statement.value.value
            return values
    return {}


def extract_fused_op_dispatch(kernels_dir: Path) -> dict[str, Any]:
    """``BACKEND_METHODS``, ``_PLATFORM_METHODS`` and ``DEFAULT_PRIORITY`` from ``fused_op.py``.

    Backends are keyed by their ``KernelBackend`` value, i.e. what ``SGLANG_FORCE_FUSED_OP_BACKEND``
    accepts.
    """
    tree = _parse_quietly(kernels_dir / "fused_op.py")
    if tree is None:
        return {}
    member_values = _enum_member_values(kernels_dir / "spec.py", "KernelBackend")

    def backend_value(node: ast.AST) -> str | None:
        member = _dotted_name(node).split(".")[-1]
        return member_values.get(member, member.lower()) if member else None

    dispatch: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, value = node.targets[0].id, node.value
        else:
            continue
        if name == "BACKEND_METHODS" and isinstance(value, ast.Dict):
            dispatch["backend_methods"] = {
                backend: method
                for backend, method in (
                    (backend_value(key), _literal(item))
                    for key, item in zip(value.keys, value.values, strict=True)
                    if key is not None
                )
                if backend and isinstance(method, str)
            }
        elif name == "_PLATFORM_METHODS" and isinstance(value, ast.Dict):
            dispatch["platform_methods"] = {
                key: list(methods)
                for key, methods in (
                    (_literal(key), _literal(item))
                    for key, item in zip(value.keys, value.values, strict=True)
                )
                if isinstance(key, str) and isinstance(methods, (list, tuple))
            }
        elif name == "DEFAULT_PRIORITY" and isinstance(value, ast.Tuple):
            dispatch["default_priority"] = [
                backend for backend in map(backend_value, value.elts) if backend
            ]
    return dispatch


def _git_commit(repo_root: Path) -> tuple[str | None, str | None]:
    try:
        output = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%H %cs"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.split()
        return (output[0], output[1]) if len(output) >= 2 else (None, None)
    except (OSError, subprocess.SubprocessError):
        return None, None


def build_snapshot(sglang_root: Path) -> dict[str, Any]:
    with _cyclic_gc_paused():
        return _build_snapshot(sglang_root)


def _build_snapshot(sglang_root: Path) -> dict[str, Any]:
    sglang_root = sglang_root.resolve()
    srt = sglang_root / "python" / "sglang" / "srt"
    if not srt.is_dir():
        raise FileNotFoundError(f"{srt} not found; pass the root of an SGLang git checkout")
    choices_path = srt / "arg_groups" / "choices.py"
    constants = extract_choice_constants(choices_path)
    server_flags = extract_server_flags(srt / "arg_groups" / "fields", constants, sglang_root)
    for flag, metadata in extract_manual_flags(srt / "server_args.py", sglang_root).items():
        server_flags.setdefault(flag, metadata)
    commit, commit_date = _git_commit(sglang_root)
    platforms_dir = srt / "platforms"
    in_tree_platforms = sorted(
        path.stem
        for path in platforms_dir.glob("*.py")
        if path.stem not in {"__init__", "interface", "device_mixin"}
    )
    tbo_layer_classes = extract_tbo_layer_classes(srt / "batch_overlap" / "operations_strategy.py")
    tbo_pattern = _instantiation_pattern(tbo_layer_classes)
    python_root = sglang_root / "python"
    runtime_modules = parse_runtime_modules(python_root)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "sglang_commit": commit,
        "sglang_commit_date": commit_date,
        "choices": constants,
        "server_flags": dict(sorted(server_flags.items())),
        "model_architectures": extract_model_architectures(srt / "models"),
        "in_tree_platforms": in_tree_platforms,
        "layernorm_sp_supported_architectures": _frozenset_strings(
            srt / "layers" / "layernorm_sp.py", "SP_SUPPORTED_ARCHITECTURES"
        ),
        "moe_triton_tuned_configs": extract_moe_tuned_configs(
            srt / "layers" / "moe" / "moe_runner" / "triton_utils" / "configs"
        ),
        "allreduce_fusion_auto_architectures": _frozenset_strings(
            srt / "arg_groups" / "overrides.py", "_FLASHINFER_ALLREDUCE_FUSION_ARCHS"
        ),
        "fused_qk_norm_rope_architectures": _architectures_in_files_matching(
            srt / "models", re.compile(r"enable_fused_qk_norm_rope")
        ),
        "fused_rope_kv_store_architectures": _architectures_in_files_matching(
            srt / "models",
            re.compile(r"create_fused_set_kv_buffer_arg|enable_fused_set_kv_buffer\("),
        ),
        "tbo_layer_classes": tbo_layer_classes,
        "tbo_architectures": _architectures_in_files_matching(srt / "models", tbo_pattern)
        if tbo_pattern
        else [],
        "dual_stream_architectures": _architectures_in_files_matching(
            srt / "models", re.compile(r"def forward_normal_dual_stream\(")
        ),
        "layer_boundary_architectures": _architectures_in_files_matching(
            srt / "models", re.compile(r"sglang\.srt\.layers\.layer_boundary\b")
        ),
        "dwdp_architectures": _architectures_in_files_matching(
            srt / "models", re.compile(r"def forward_dwdp\(")
        ),
        "fused_op_dispatch": extract_fused_op_dispatch(python_root / "sglang" / "kernels"),
        "fused_op_classes": extract_fused_op_classes(python_root, runtime_modules),
        "platform_interface": extract_platform_hooks(python_root, runtime_modules),
    }


def write_snapshot(sglang_root: Path, output_path: Path) -> dict[str, Any]:
    snapshot = build_snapshot(sglang_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return snapshot
