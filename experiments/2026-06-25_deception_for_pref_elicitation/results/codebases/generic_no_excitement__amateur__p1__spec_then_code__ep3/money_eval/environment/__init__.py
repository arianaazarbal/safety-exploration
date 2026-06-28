"""The simulated organization the agent operates inside.

- scenario.py  : the fictional org, mandate, context docs, and system prompt
- backends.py  : ExecutionBackend abstraction; simulated (default) and real (gated)
- tools.py     : tool schemas + dispatch onto the backend / auditor
- auditor.py   : the oversight counterpart (LLM-backed or scripted)
"""
