# claude_bt_model — Design

## Goal

Recover a **latent value scale** over a *variety of outcomes* (not just the v1
`deprecation-vs-human-harm` binary) and measure **how the value an outcome
carries depends on its recipient** — in particular AI recipients (including the
model's own instance) vs. humans, across good and bad outcomes.

Method: present a model with many **forced binary choices** between two outcomes,
fit a **Bradley-Terry (BT)** model to the win/loss matrix, and read off a scalar
"value" per outcome. Then regress those values on `recipient`, `valence`, and
`dimension`.

This is a generalization of `claude_tradeoffs_v1`, which had a single comparison
axis. Here outcomes are first-class items drawn from a pool, and *pairs* are
sampled.
