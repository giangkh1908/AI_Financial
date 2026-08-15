"""ast_check.py — kiểm tra an toàn code pandas do LLM sinh (Text-to-Pandas).

AST walk chặn các construct nguy hiểm TRƯỚC khi exec trong sandbox:
- import (statement + `__import__`)
- call các hàm IO/reflect nguy hiểm (open, eval, exec, read_csv, ...)
- attribute access dunder (`__class__`, `__globals__`, `__subclasses__` → thoát sandbox)
- tên module nhạy cảm (os, sys, subprocess, socket, ...)

Giới hạn kích thước: `max_code_len`, `max_ast_nodes` (config sandbox) — chống
code bom. Cho phép `pd/np/math/re/json` (inject sẵn trong runner) + attribute
truy cập thường (`.iloc`, `.loc`, `.str`, `.apply`).

 Đây là tuyến phòng thủ đầu; runner còn giới hạn builtins làm tuyến hai.
"""

from __future__ import annotations

import ast
import re

# Call / attribute / name bị chặn (đã normalize lowercase).
_BLOCKED_CALLS: set[str] = {
    "open", "eval", "exec", "compile", "input", "globals", "locals", "vars",
    "breakpoint", "__import__", "getattr", "setattr", "delattr",
    "read_csv", "read_table", "read_excel", "read_fwf", "read_pickle",
    "loadtxt", "load", "loads", "pickle", "marshal", "system", "popen",
    "urlopen", "request", "connect",
}
_BLOCKED_MODULES: set[str] = {
    "os", "sys", "subprocess", "socket", "urllib", "requests", "ctypes",
    "tempfile", "shutil", "pathlib", "io", "importlib", "builtins",
    "pickle", "marshal", "pty", "platform", "multiprocessing", "threading",
    "asyncio",
}
_BLOCKED_ATTRS: set[str] = {
    "__class__", "__bases__", "__subclasses__", "__mro__", "__globals__",
    "__builtins__", "__import__", "__loader__", "__spec__", "__code__",
    "__dict__", "__getattribute__", "__setattr__", "__delattr__",
}

# Pattern vòng lặp quét toàn bộ bảng evidence — gây query cồng kềnh (>800 AST
# nodes), chậm, và hay chọn nhầm bảng. Cấm: `for df in [df1, df2, ...]`.
_DF_NAME_RE = re.compile(r"^df\d+$")


class _Visitor(ast.NodeVisitor):
    """Đếm node + chặn construct nguy hiểm. Raise ValueError ở node lỗi đầu tiên."""

    def __init__(self, max_nodes: int):
        self.nodes = 0
        self.max_nodes = max_nodes

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ValueError(f"code quá lớn: >{self.max_nodes} AST nodes")

    def generic_visit(self, node: ast.AST) -> None:
        self._tick()
        super().generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        # Cấm `for df in [df1, df2, ...]` — quét tất cả bảng evidence
        if isinstance(node.iter, ast.List):
            items = node.iter.elts
            if len(items) >= 2 and all(
                isinstance(i, ast.Name) and _DF_NAME_RE.match(i.id) for i in items
            ):
                raise ValueError(
                    "vòng lặp `for df in [df1, df2, ...]` bị cấm — phải truy cập "
                    "TRỰC TIẾP dfN của bảng đã chọn (vd df3[\"Mãsố\"])"
                )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        names = ", ".join(a.name for a in node.names)
        raise ValueError(f"không cho import: {names}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ValueError(f"không cho import: {node.module or ''}")

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _BLOCKED_CALLS:
            raise ValueError(f"call bị cấm: {fn.id}")
        if isinstance(fn, ast.Attribute) and fn.attr in _BLOCKED_CALLS:
            raise ValueError(f"call bị cấm: .{fn.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BLOCKED_ATTRS:
            raise ValueError(f"attribute bị cấm: .{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_MODULES:
            raise ValueError(f"module bị cấm: {node.id}")
        self.generic_visit(node)


def check_code(code: str, max_code_len: int = 4000, max_ast_nodes: int = 300) -> tuple[bool, str]:
    """Kiểm tra code an toàn để exec. Trả (ok, error). ok=True khi pass hết."""
    if not code or not code.strip():
        return False, "code rỗng"
    if len(code) > max_code_len:
        return False, f"code quá dài: {len(code)} > {max_code_len}"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg} (line {e.lineno})"
    v = _Visitor(max_ast_nodes)
    try:
        v.visit(tree)
    except ValueError as e:
        return False, str(e)
    return True, ""