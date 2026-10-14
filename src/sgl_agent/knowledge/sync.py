"""Build a capability snapshot from an SGLang source checkout.

Everything here is static analysis (``ast``): SGLang itself is never imported, so the
snapshot can be regenerated on a machine without GPUs or SGLang's dependencies.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
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
    }


def write_snapshot(sglang_root: Path, output_path: Path) -> dict[str, Any]:
    snapshot = build_snapshot(sglang_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return snapshot
