import pytest

from vm import VM, VMError


def run(program):
    return VM().run(program)


def test_push_returns_top():
    assert run([("PUSH", 7)]) == 7


def test_add():
    assert run([("PUSH", 2), ("PUSH", 3), ("ADD",)]) == 5


def test_sub_order():
    assert run([("PUSH", 10), ("PUSH", 4), ("SUB",)]) == 6


def test_mul():
    assert run([("PUSH", 6), ("PUSH", 7), ("MUL",)]) == 42


def test_div_order():
    assert run([("PUSH", 20), ("PUSH", 5), ("DIV",)]) == 4


def test_mod_order():
    assert run([("PUSH", 17), ("PUSH", 5), ("MOD",)]) == 2


def test_div_by_zero_raises():
    with pytest.raises(VMError):
        run([("PUSH", 3), ("PUSH", 0), ("DIV",)])


def test_dup_and_swap():
    assert run([("PUSH", 2), ("DUP",), ("ADD",)]) == 4
    assert run([("PUSH", 9), ("PUSH", 1), ("SWAP",), ("SUB",)]) == -8


def test_jz_skips_when_nonzero():
    program = [
        ("PUSH", 1),
        ("JZ", 4),
        ("PUSH", 100),
        ("HALT",),
        ("PUSH", 200),
    ]
    assert run(program) == 100


def test_countdown_loop():
    program = [
        ("PUSH", 5),
        ("PUSH", 1),
        ("SUB",),
        ("DUP",),
        ("JZ", 7),
        ("JMP", 1),
        ("HALT",),
        ("PUSH", 0),
    ]
    assert run(program) == 0


def test_stack_underflow_raises():
    with pytest.raises(VMError):
        run([("ADD",)])


def test_empty_program_returns_none():
    assert run([]) is None
