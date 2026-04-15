"""Pretty-printing helpers.

`pp(obj, json=True)` emits the project's canonical JSON save format (see the example file
`multilayer_network.json`): scalar-only lists stay on one line, lists-of-lists expand with
two-space indent. `pp(obj, json=False)` falls through to `pprint.pp`.

This module keeps a `json` kwarg on a `pp` callable so user code can write::

    from src import pp
    pp(state, json=True)
"""

from __future__ import annotations

import json as _json
import pprint as _pprint
import sys
from typing import IO, Any


def pp(obj: Any, *args, json: bool = False, stream: IO | None = None, **kwargs) -> None:
    """Pretty-print ``obj``.

    When ``json=True`` the output matches the project's JSON save format exactly. Otherwise
    delegates to :func:`pprint.pp` with the remaining arguments.
    """
    if not json:
        return _pprint.pp(obj, *args, stream=stream, **kwargs)
    (stream or sys.stdout).write(to_json_string(obj) + "\n")


def to_json_string(obj: Any, indent: int = 2) -> str:
    """Serialize ``obj`` as JSON, formatted like the project's save files.

    Rules:
      - Lists whose elements are all scalars (numbers / bools / strings / None) stay on one line.
      - Lists containing lists, tuples, or dicts expand, one element per line, indented by
        ``indent`` spaces per level.
      - Dicts always expand, with keys emitted in insertion order.
    """
    return _emit(obj, 0, indent)


def _emit(value: Any, cur_indent: int, step: int) -> str:
    pad = " " * cur_indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        child_pad = " " * (cur_indent + step)
        parts = [f'{child_pad}"{k}": {_emit(v, cur_indent + step, step)}' for k, v in value.items()]
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        if all(_is_scalar(x) for x in value):
            return "[" + ", ".join(_scalar_repr(x) for x in value) + "]"
        child_pad = " " * (cur_indent + step)
        parts = [child_pad + _emit(x, cur_indent + step, step) for x in value]
        return "[\n" + ",\n".join(parts) + "\n" + pad + "]"
    return _scalar_repr(value)


def _is_scalar(x: Any) -> bool:
    # NumPy scalars expose `.item()`; treat them as scalars by dispatching through _scalar_repr.
    if hasattr(x, "item") and not isinstance(x, (list, tuple, dict)):
        try:
            x = x.item()
        except Exception:
            return False
    return isinstance(x, (int, float, bool, str)) or x is None


def _scalar_repr(x: Any) -> str:
    if hasattr(x, "item") and not isinstance(x, (int, float, bool, str)) and x is not None:
        try:
            x = x.item()
        except Exception:
            pass
    return _json.dumps(x)
