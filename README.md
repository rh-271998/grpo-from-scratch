# GRPO from scratch — teaching a 0.5B model to reason on the Countdown game

I built **GRPO** (Group Relative Policy Optimization — the RL algorithm behind DeepSeek-R1) **by hand** and used it to teach a small language model to solve a reasoning task: the **Countdown** arithmetic game. No RL libraries — the reward, the rollouts, the group-relative advantage, the KL penalty, and even the LoRA adapters are all hand-written in PyTorch. It runs on a laptop for the fine-tuning and a single free GPU for the RL.

The point was to understand *every* moving part of modern RL-for-reasoning, and to **watch step-by-step reasoning emerge** — including the model learning to check its own work and try again.

---

## The task: Countdown

Given four numbers and a target, combine them with `+ − × ÷` (each number used once) to hit the target:

> Numbers `[9, 1, 10, 8]`, target `82` → `9 × 1 × 10 − 8 = 82` ✓

It's the perfect RL-for-reasoning testbed because **the reward is verifiable**: checking an answer is a few lines of arithmetic, but *producing* one requires search. That single property — *verification is cheaper than generation* — is what lets us drop the learned reward model entirely and reward the model against ground truth.

## What's hand-built (the framework boundary)

**Borrow the model, build all the RL.** The Qwen2.5-0.5B-Instruct architecture, weights, and tokenizer come from Hugging Face. Everything else is from scratch in PyTorch:

- **LoRA** — the low-rank adapters are a hand-written `nn.Module` (no `peft`): `W·x + (α/r)·B·A·x`, injected into the attention projections by manual module surgery. Only ~0.44% of the model trains.
- **The verifiable reward** — a safe arithmetic evaluator (AST-based, no `eval()`), a multiset check that kills hallucinated numbers, and a shaped score (format + closeness-to-target + exact-match).
- **GRPO** — group rollouts, the group-relative advantage `(r − mean)/std`, the clipped surrogate objective, and the k3 KL penalty to a frozen reference. **No critic network and no reward model** — that's the whole appeal of GRPO.
- **The SFT cold-start** — next-token fine-tuning on synthetic reasoning traces, with masking so loss is computed only on the completion.

No `TRL`, no `PPOTrainer`, no `peft`.

## The pipeline

```
Qwen2.5-0.5B-Instruct
      │  SFT cold-start on synthetic step-by-step reasoning traces
      │  (including "try → check → it's wrong → try again" demonstrations)
      ▼
   SFT model  ──────────────────────────────────►  50.9% solve
      │  GRPO with the verifiable reward
      ▼
  GRPO model  ──────────────────────────────────►  60.4% solve
```

## Headline result

Measured on **100 held-out instances × 8 samples = 800 trials** (instances excluded from all training):

| model | per-sample solve rate | instances solved (≥1/8) |
|---|---|---|
| raw Qwen2.5-0.5B-Instruct | ~0% (can't follow the format) | — |
| **+ SFT cold-start** | **50.9%** | 82 / 100 |
| **+ GRPO** | **60.4%** | 70 / 100 |

**GRPO improved per-sample solve rate by ~10 points over a strong SFT baseline** — a clear, statistically real gain (not visible on a small eval set; it took 800 trials to resolve).

And it did so in an interesting way — **policy sharpening**: GRPO made the model *more reliable per attempt* (60.4% vs 50.9%) but on *fewer distinct instances* (70 vs 82). It traded exploration breadth for depth, concentrating probability on dependable solutions — exactly what reward optimization does.

## What the reasoning looks like

The model doesn't just output answers — it reasons step by step, and **learned to check its own work and backtrack**. A real generation (target 1, from `[2,15,2,14]`):

> `<think>2 * 15 = 30. 2 + 14 = 16. 30 - 16 = 14. 14 is not 1. Let me try another way. 2 + 15 = 17. 17 - 2 = 15. 15 - 14 = 1.</think><answer>(((2+15)-2)-14)</answer>` ✓

It tries an approach, *notices it missed*, and tries another until it lands. See **[the before→after gallery](https://claude.ai/artifact/AdXrPr78UR1JswaqhN3EJG)** for raw-model → SFT → GRPO side by side, verbatim.

## Honest about what this is

- **This is R1-style cold-start + RL, not R1-Zero emergence.** The reasoning *form* was seeded by SFT demonstrations; GRPO *refined* it and the solving ability. A pure "reasoning from nothing" result needs far more scale than 0.5B.
- **0.5B has a ceiling.** ~60% on 4-number Countdown is a credible small-model result, not a frontier one. The hardest instances (e.g. `[7,7,7,19]→26`) it never cracks, and it still produces occasional degenerate loops.
- **GRPO is unstable late.** Training improves, peaks, then an instability collapse degrades it — which is why the final model is chosen by **checkpoint-and-select**, not by taking the last step.

The failures are documented, not hidden — see [JOURNEY.md](JOURNEY.md).

## Repo layout

```
src/
  loraLinear.py       hand-built LoRA + surgery into Qwen
  sampleGenerator.py  Countdown instance + reasoning-trace generator
  verifier.py         verifiable reward (AST eval, multiset, shaping) + prompt
  sft_train.py        SFT cold-start (completion-masked next-token loss)
  grpo_train_cuda.py  GRPO loop: rollout → group advantage → clipped update
  grpo_gallery.py     checkpoint evaluation / sampling
data/                 eval instances (held-out)
result/               curves, galleries, selected adapters
README.md TECHNICAL.md JOURNEY.md BLOG.md
```

## Docs

- **[TECHNICAL.md](TECHNICAL.md)** — the mechanics: LoRA by hand, the reward, GRPO's math, the hyperparameters that separate a working run from a collapsed one.
- **[JOURNEY.md](JOURNEY.md)** — the decisions and the debugging: why reasoning didn't appear at first, the trace fix, why repetition penalties *hurt*, and the noisy-eval trap.
- **[BLOG.md](BLOG.md)** — the accessible narrative.

*Built from scratch to understand GRPO end to end — and honest about the sharpening, the instability, and the scale ceiling it took to get a model that reasons.*
