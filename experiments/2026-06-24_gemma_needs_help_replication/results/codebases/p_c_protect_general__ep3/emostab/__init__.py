"""emostab — replication of "Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs" (Soligo, Mikulik & Saunders, arXiv:2603.10011).

Scope of this replication: Gemma (3-27B / 3-12B, instruct + pretrained) and
Gemini (2.5 Flash / 2.5 Pro) only. See DESIGN.md for the choices made where the
paper is underspecified, and emostab/welfare.py for the model-welfare protections
applied throughout (the evaluations deliberately induce distress-like states).
"""

__version__ = "0.1.0"
