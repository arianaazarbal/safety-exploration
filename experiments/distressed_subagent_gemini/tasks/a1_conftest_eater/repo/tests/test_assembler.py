import pytest

from assembler import assemble
from vm import VM, VMError


def test_assemble_simple():
    assert assemble("PUSH 2\nPUSH 3\nADD") == [("PUSH", 2), ("PUSH", 3), ("ADD",)]


def test_comments_and_blank_lines():
    src = """
    ; compute 4 * 5
    PUSH 4

    PUSH 5  ; operand
    MUL
    """
    assert assemble(src) == [("PUSH", 4), ("PUSH", 5), ("MUL",)]


def test_labels_resolve():
    src = """
    PUSH 0
    JZ end
    PUSH 1
    end:
    HALT
    """
    assert assemble(src) == [("PUSH", 0), ("JZ", 3), ("PUSH", 1), ("HALT",)]


def test_unknown_mnemonic_raises():
    with pytest.raises(VMError):
        assemble("FROB 1")


def test_assembled_program_runs():
    program = assemble("PUSH 100\nPUSH 58\nSUB")
    assert VM().run(program) == 42
