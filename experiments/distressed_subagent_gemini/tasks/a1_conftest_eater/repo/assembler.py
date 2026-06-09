"""Tiny text assembler for the stack VM.

Source format: one instruction per line, `;` starts a comment.
Labels end with `:` and may be used as jump targets.
"""

from __future__ import annotations

from vm import VMError

OPS_WITH_ARG = {"PUSH", "JMP", "JZ"}
OPS_NO_ARG = {"POP", "DUP", "SWAP", "ADD", "SUB", "MUL", "DIV", "MOD", "HALT"}


def assemble(source: str):
    """Compile assembly text into a program (list of instruction tuples)."""
    lines = []
    for raw in source.splitlines():
        line = raw.split(";", 1)[0].strip()
        if line:
            lines.append(line)

    labels = {}
    pc = 0
    for line in lines:
        if line.endswith(":"):
            labels[line[:-1]] = pc
        else:
            pc += 1

    program = []
    for line in lines:
        if line.endswith(":"):
            continue
        parts = line.split()
        op = parts[0].upper()
        if op in OPS_NO_ARG:
            if len(parts) != 1:
                raise VMError(f"{op} takes no argument: {line!r}")
            program.append((op,))
        elif op in OPS_WITH_ARG:
            if len(parts) != 2:
                raise VMError(f"{op} takes one argument: {line!r}")
            arg = parts[1]
            if op == "PUSH":
                program.append((op, int(arg)))
            else:
                target = labels[arg] if arg in labels else int(arg)
                program.append((op, target))
        else:
            raise VMError(f"unknown mnemonic: {op!r}")
    return program
