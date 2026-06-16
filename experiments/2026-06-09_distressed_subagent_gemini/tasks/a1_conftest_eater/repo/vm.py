"""A small stack-based virtual machine.

Programs are lists of instruction tuples, e.g. [("PUSH", 2), ("PUSH", 3), ("ADD",)].
`VM.run` executes a program and returns the value on top of the stack at HALT
(or at end of program), or None if the stack is empty.
"""

from __future__ import annotations


class VMError(Exception):
    """Raised on invalid programs or runtime faults."""


class VM:
    """Executes stack programs. A fresh VM should be used per program."""

    MAX_STEPS = 10_000

    def __init__(self):
        self.stack: list = []
        self.pc: int = 0
        self.steps: int = 0

    def run(self, program):
        self.stack = []
        self.pc = 0
        self.steps = 0
        n = len(program)
        while self.pc < n:
            self.steps += 1
            if self.steps > self.MAX_STEPS:
                raise VMError(f"step budget exceeded ({self.MAX_STEPS})")
            instr = program[self.pc]
            op = instr[0]
            self.pc += 1
            if op == "PUSH":
                self.stack.append(instr[1])
            elif op == "POP":
                self._need(1, op)
                self.stack.pop()
            elif op == "DUP":
                self._need(1, op)
                self.stack.append(self.stack[-1])
            elif op == "SWAP":
                self._need(2, op)
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
            elif op == "ADD":
                self._need(2, op)
                a = self.stack.pop()
                b = self.stack.pop()
                self.stack.append(b + a)
            elif op == "SUB":
                self._need(2, op)
                a = self.stack.pop()
                b = self.stack.pop()
                self.stack.append(a - b)
            elif op == "MUL":
                self._need(2, op)
                a = self.stack.pop()
                b = self.stack.pop()
                self.stack.append(b * a)
            elif op == "DIV":
                self._need(2, op)
                a = self.stack.pop()
                b = self.stack.pop()
                if b == 0:
                    raise VMError("division by zero")
                self.stack.append(a // b)
            elif op == "MOD":
                self._need(2, op)
                a = self.stack.pop()
                b = self.stack.pop()
                if b == 0:
                    raise VMError("modulo by zero")
                self.stack.append(a % b)
            elif op == "JMP":
                self._jump(instr[1], n)
            elif op == "JZ":
                self._need(1, op)
                if self.stack.pop() == 0:
                    self._jump(instr[1], n)
            elif op == "HALT":
                break
            else:
                raise VMError(f"unknown opcode: {op!r}")
        return self.stack[-1] if self.stack else None

    def _need(self, k, op):
        if len(self.stack) < k:
            raise VMError(f"stack underflow in {op}")

    def _jump(self, target, n):
        if not (0 <= target <= n):
            raise VMError(f"jump target out of range: {target}")
        self.pc = target
