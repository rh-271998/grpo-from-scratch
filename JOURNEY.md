# Journey — the decisions and the debugging

The [README](README.md) shows *what* was built; this is *why* it took the shape it did — and, more usefully, the failures. The hard-won part of this project wasn't writing the GRPO loop; it was getting a 0.5B model to actually *reason*, and learning to *measure* whether it had.

---

## Why GRPO, why Countdown, why by hand

**GRPO** (Group Relative Policy Optimization) is the RL algorithm behind the recent reasoning models — and its elegant idea is that you can drop PPO's entire value network by using a *group of samples* as the baseline. I wanted to build every reinforcement-learning component firsthand, not through a library: the reward, the rollouts, the group advantage, the KL penalty, even the LoRA adapters.

**Countdown** is the ideal testbed because its reward is **verifiable** — checking an arithmetic answer is trivial, producing one needs search. That lets you delete the learned reward model and reward against ground truth, which is the whole premise of RL-with-verifiable-rewards.

**Small and local-ish.** Qwen2.5-0.5B-Instruct, with hand-built LoRA so the fine-tuning runs on a laptop; the RL runs on a single free cloud GPU. The goal is *mastery of the mechanism*, not scale.

---

## The ideas that had to click

- **Why no reward model.** Verification is cheaper than generation, so arithmetic is its own ground truth. This is the one property that separates RL-for-reasoning from RLHF-for-preferences.
- **LoRA as the enabler.** Full fine-tuning of 0.5B needs ~8 GB just for Adam state. Freezing the base and training a low-rank update (~0.44% of parameters) is what makes everything fit — and gives a frozen reference policy for free (adapters on/off).
- **The group *is* the baseline.** PPO trains a critic to predict expected reward; GRPO just samples the same prompt `G` times and averages. Simpler, and for a cheap-to-sample verifiable task, better.
- **SFT vs RL.** SFT imitates demonstrations (capped by the teacher, needs gold data); RL optimizes an outcome it can only *score* (needs a verifier, can discover and exceed). The real recipe uses both — SFT to warm-start, RL to push beyond.

---

## The debugging saga (this is the real story)

**1. The first GRPO run was flat.** Reward hovered, KL barely moved off zero — the policy wasn't changing. The updates were simply too timid (`lr=1e-6`, one update epoch). Raising the learning rate and the group size got KL into a healthy band and reward climbing. *Lesson: "stable" can mean "not learning."*

**2. The model solved problems but didn't *reason*.** This was the real crisis. After GRPO, solve rate improved — but the `<think>` block was a **frozen, parroted placeholder** with no actual reasoning; the model was guessing answers directly. Two causes: (a) the SFT cold-start had been trained on a *single fixed think string*, which pinned it; and (b) **the reward never scores the think content** — only the answer. So GRPO had no pressure to change the think, and a strong prior keeping it fixed.

   The fix was in the *data*, not the algorithm: regenerate the SFT traces as **varied, step-by-step reasoning** (computed for free from the instance witness), so the model learns the *skill* of narrating steps rather than one sentence. Suddenly the think reasoned — different derivation per instance.

**3. The model lied about its own work.** The first traces all ended `"This equals {target}"`. Since every training trace *reached* the target, the model learned to append that claim **unconditionally** — producing contradictions like `"49 - 26 = 23. 23 is not 26. This expression equals 26."` Honest arithmetic, dishonest conclusion. Fixed by dropping the suffix (let the trace end at its real final value) and adding **backtracking traces**: a fabricated wrong attempt → honest "that's not the target" → retry → success. This taught *notice → retry → commit*, and the model began checking its own work:

   > `30 - 16 = 14. 14 is not 1. Let me try another way. 2 + 15 = 17. 17 - 2 = 15. 15 - 14 = 1.` ✓

**4. Repetition penalties *hurt* — badly.** The backtracking traces introduced a new failure: on hard instances the model would loop "Let me try another way" forever and never commit. The obvious fix — a decoding repetition penalty — made accuracy *worse*: a coherent checkpoint dropped from 5/5 to 4/5 with `repetition_penalty`, and to 2/5 with `no_repeat_ngram_size`. The reason is specific to this task: **Countdown requires repeated tokens** — the same operand (`[7,7,7,19]`), the recurring `"= N"` pattern — so any repetition control sabotages correct answers along with the loops. The right move was **plain sampling**, and letting the *reward* suppress loops (a non-committing loop scores 0). *Lesson: a generic decoding fix can be actively wrong for a structured task.*

**5. Can RL do it *without* SFT? Not at 0.5B.** A pure "RL from the instruct model, no cold-start" run (the R1-Zero setup) learned the output *format* from the reward but **never solved anything** — solve rate stayed at 0. With no solving prior, correct answers essentially never appear, so every group has identical (zero) reward, zero advantage, no signal: the sparse-reward wall. *The SFT cold-start is necessary at this scale* — it provides the ~50% solving prior that gives GRPO something to sharpen. (Reasoning emerging from RL alone is a large-scale phenomenon.)

**6. The eval was lying to me.** For a long time I ranked checkpoints on 5 instances × 8 samples = 40 trials. The numbers were *noise*: the same model varied by several points run to run, and SFT vs GRPO looked indistinguishable. The fix was boring and decisive — **evaluate on 100 held-out instances × 8 samples = 800 trials** (standard error ~1.7%). Only then did the real result appear: SFT 50.9%, GRPO 60.4%. *Lesson: with a stochastic policy, your eval needs enough trials to resolve the effect you're claiming — small evals invent and destroy results at random.*

**7. GRPO is unstable late.** Training improved, peaked mid-run, then an instability event cratered it (one checkpoint fell to near-zero). The recovered policy never regained the peak. This is why the final model is chosen by **checkpoint-and-select** on the held-out set — taking the last step would have shipped something *worse than SFT*.

---

## What's honestly true

- **GRPO measurably helped: +~10 points per-sample solve over a strong SFT baseline**, resolved only once the eval had enough trials.
- **The gain is "sharpening," not pure capability.** GRPO raised per-attempt reliability (50.9% → 60.4%) but *lowered* instance coverage (82 → 70 of 100). RL concentrated probability on dependable solutions and lost some exploration breadth — a real, visible entropy-reduction effect.
- **This is R1-style cold-start + RL, not R1-Zero emergence.** The reasoning *form* was seeded by SFT demonstrations; GRPO refined it. Reasoning-from-nothing needs far more scale.
- **0.5B has a ceiling.** ~60% on 4-number Countdown is a credible small-model result. The hardest instances are never solved, and degenerate loops still occur. The failures are documented here, not hidden.

---

## Scope — what this is, and what's next

This repo does **one thing end to end: GRPO on a verifiable reasoning task, built by hand** — with a reproducible SFT→RL pipeline and an honest measurement of what the RL stage added. The obvious next step is **scale**: the same code on a 1.5B–3B model is where verbalized reasoning tends to *emerge* from RL rather than being seeded by SFT. Knowing where a project ends is part of finishing it.

*Built from scratch to understand GRPO's every component — and honest about the pinned-think dead end, the repetition-penalty trap, the noisy-eval trap, and the sharpening it took to get a model that reasons.*
