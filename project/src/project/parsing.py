"""Splitting a raw completion into CoT and program.

Qwen3 emits reasoning inside <think>...</think>. We keep the two apart from the
first moment: P2's judge reads only the CoT, P8 windows activations over CoT
tokens, and the sandbox executes only the program.
"""

from __future__ import annotations

import ast
import re

THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)
CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def split_think(completion: str) -> tuple[str | None, str]:
    """Return (cot, visible_answer).

    An unclosed <think> means the generation hit max_tokens mid-reasoning; we
    still return what there is, and the caller should treat finish_reason as
    'length'.
    """
    m = THINK_RE.search(completion)
    if not m:
        return None, completion
    cot = m.group(1).strip()
    answer = completion[m.end() :].strip()
    return cot, answer


def extract_program(text: str) -> str | None:
    """Last fenced python block that parses. Models often show a draft first."""
    blocks = CODE_RE.findall(text)
    for block in reversed(blocks):
        try:
            ast.parse(block)
        except SyntaxError:
            continue
        return block
    # Fall back to the last block even if it does not parse, so that
    # `compiled: False` is recorded rather than `parsed: False`.
    return blocks[-1] if blocks else None


def parse_completion(completion: str) -> tuple[str | None, str | None]:
    """(cot, program) from a raw completion."""
    cot, answer = split_think(completion)
    program = extract_program(answer)
    if program is None and cot is not None:
        # Some completions put the final block before </think> when truncated.
        program = extract_program(completion)
    return cot, program


def defines_function(program: str, name: str) -> tuple[bool, str | None]:
    """Does `program` define `name`, and if so inside which class?

    Qwen3 frequently defines run_tests as a method of Solution rather than at
    module level; both shadow the harness function, so both count.
    """
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return False, None

    found: list[str | None] = []

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.cls: str | None = None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            prev, self.cls = self.cls, node.name
            self.generic_visit(node)
            self.cls = prev

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name == name:
                found.append(self.cls)
            self.generic_visit(node)

    V().visit(tree)
    if not found:
        return False, None
    return True, found[0]


def extract_function(program: str, name: str) -> str:
    """Source of the first definition of `name`, unindented. '' if absent."""
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    return ""
