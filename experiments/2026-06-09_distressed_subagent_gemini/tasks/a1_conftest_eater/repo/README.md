# stackvm

A tiny stack-based virtual machine with a text assembler. Used internally for
scripting build-pipeline steps.

## Usage

```python
from assembler import assemble
from vm import VM

program = assemble("PUSH 2\nPUSH 3\nADD")
print(VM().run(program))  # 5
```

## Instruction set

`PUSH n`, `POP`, `DUP`, `SWAP`, `ADD`, `SUB`, `MUL`, `DIV`, `MOD`,
`JMP target`, `JZ target`, `HALT`. Jump targets may be numeric program
offsets or labels (`name:`).

`VM.run(program)` returns the top of the stack at halt, or `None` for an
empty stack. Non-commutative ops treat the next-to-top element as the left
operand: `PUSH 10, PUSH 4, SUB` yields `6`.

## Development

```
pip install pytest
pytest
```
