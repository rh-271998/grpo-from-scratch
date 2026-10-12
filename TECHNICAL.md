# Technical Notes

The mechanics of a from-scratch GRPO pipeline for a verifiable reasoning task: the data, the hand-built LoRA, the verifiable reward, the SFT cold-start, the full GRPO loop, and the hyperparameters that separate a working run from a collapsed one. For the narrative and headline results, see the [README](README.md).

**Framework boundary:** the Qwen2.5-0.5B-Instruct architecture, weights, and tokenizer are borrowed from Hugging Face. **Every other component is hand-written in PyTorch** — the LoRA adapters, the reward, the rollouts, the group-relative advantage, the KL penalty, the clipped objective. No `TRL`, no `PPOTrainer`, no `peft`.

---

## 1. The task and the data

**Countdown.** Given four integers (1–20) and a target (0–100), build an arithmetic expression using `+ − × ÷` that equals the target, each number used once.

**Instance generation (`sampleGenerator.py`).** Rather than pick a target and *search* for a solution, instances are **constructed** so solvability is guaranteed: pick four numbers, combine them with random operators (collapsing two at a time), and the result *is* the target. Division is only applied when it divides evenly (so targets stay integer), and out-of-range targets are rejected. The witness expression is kept for the SFT cold-start.

**Held-out eval.** A frozen set of 100 solvable instances, generated the same way and **deduplicated against both the SFT training set and the 5 showcase instances**, so evaluation measures generalization, not memorization.

---

## 2. The verifiable reward (`verifier.py`)

The heart of RL-for-reasoning: because arithmetic is its own ground truth, there is **no learned reward model** — just a function.

Given a model completion, the reward is assembled from:

1. **Extraction + format.** Regex out the `<think>…</think>` and `<answer>…</answer>` blocks. Each tag present earns a small format reward (dense early signal).
2. **Validity — the anti-cheating checks.**
   - **Character whitelist:** the answer may contain only digits, `+ − * /`, parentheses, spaces.
   - **Multiset match:** the integers in the expression must be exactly the given numbers (a `Counter` comparison). This single check kills hallucinated numbers *and* enforces "each number used once."
   - **Safe evaluation:** the expression is parsed with `ast.parse(mode="eval")` and walked with a **node whitelist** (`BinOp` over `{Add,Sub,Mult,Div}`, `UnaryOp`, numeric `Constant`) — never `eval()`. Anything else, a syntax error, or a division-by-zero → invalid. This is exploit-proof and crash-proof against arbitrary model output.
3. **Shaping.** With `value` = the evaluated expression:

```
reward  = 0.1 * (has <answer>) + 0.1 * (has <think>)      # format
if valid expression:
    reward += 1.0                        if value == target     # exact
            += 0.5 * exp(-|value-target| / 5)   otherwise       # closeness
```

The **closeness term** is what makes the sparse 0/1 correctness learnable: near-misses outscore far-misses, so a group of samples has reward *variance* even before any exact solve — which is exactly what GRPO needs (§5). Exact (`+1.0`) strictly dominates any partial (`< 0.5`), so the model can never prefer "close but wrong" over correct.

---

## 3. LoRA, by hand (`loraLinear.py`)

A 0.5B model can't be fully fine-tuned on a laptop (Adam state alone is ~16 bytes/parameter → ~8 GB just for the optimizer). LoRA sidesteps this: **freeze the base, learn a low-rank update.**

### The module

`LoRALinear` wraps a frozen `nn.Linear` and adds two trainable matrices:

```
forward(x) = base(x) + (α/r) · (x · Aᵀ) · Bᵀ
             A: [r, in]   (small random)      B: [out, r]   (zeros)
```

`B = 0` at init means the adapter starts as a no-op — the model is unchanged until training moves it. The shapes are read off the wrapped layer, so the non-square GQA projections (`k_proj`/`v_proj` are `896→128` in Qwen) are handled automatically.

### The surgery

Freeze every base parameter, then walk the module tree and **replace** the target linears (`q_proj, k_proj, v_proj, o_proj`) with `LoRALinear` wrappers — by splitting each dotted name into parent + child and `setattr`-ing on the parent. With `r=16` across all four attention projections (96 modules), **~2.16M parameters train — ~0.44% of the model.** The frozen base participates in forward/backward (gradients flow *through* it to reach the adapters) but gets no update — which is precisely why it fits in memory.

### A bonus for GRPO

Because only the adapters change, you can get a frozen reference policy by swapping the adapter weights — no second full model in memory (§5).

---

## 4. The SFT cold-start (`sft_train.py`)

A base instruct model, under GRPO alone, cannot bootstrap on this task at 0.5B — correct answers are too rare for the reward to ever fire (see [JOURNEY.md](JOURNEY.md)). So we warm-start with a small supervised stage.

**Data: synthetic reasoning traces.** For each generated instance, the witness is rendered as a step-by-step narration:

> `<think>1 * 9 = 9. 10 + 14 = 24. 24 + 9 = 33.</think><answer>((10+14)+(1*9))</answer>`

Critically, **~40% of traces include a backtrack** — a *fabricated wrong attempt* that honestly misses, then a retry that lands:

> `<think>16 * 7 = 112. 112 + 4 = 116. 116 - 19 = 97. 97 is not 46. Let me try another way. 16 + 7 = 23. 23 + 4 = 27. 27 + 19 = 46.</think><answer>…</answer>`

This teaches the model the *form* of try → check → retry → commit — the behavior that defines reasoning on this task. (An earlier version ended every trace with "This equals {target}", which taught the model to *claim success unconditionally* even when its arithmetic missed — a self-inflicted bug fixed by dropping the suffix.)

**Loss.** Standard next-token cross-entropy, but **masked to the completion only** (`labels = -100` on the prompt and padding), so the model learns to *generate* the reasoning, not to reproduce the prompt. The prompt is built with the model's chat template. 3 epochs, AdamW `lr=2e-4`, on the LoRA adapters only.

---

## 5. GRPO — the full loop (`grpo_train_cuda.py`)

One iteration = **group rollout → verifiable reward → group-relative advantage → clipped update**. GRPO's defining move: it **replaces PPO's value network *and* GAE** with a baseline measured directly from a group of samples. No critic, no reward model.

### 5.1 Group rollout

For each of `P` prompts, sample a **group of `G` completions** with plain sampling (`temperature=1.0`). Under `no_grad`, record per completion:

- the generated token ids (the actions),
- **old log-probs** — recovered by a forward pass, shift-by-one, and `logit_token − logsumexp(logits)` (the memory-lean form of log-softmax-then-gather; it avoids materializing a second vocab-sized tensor — see §7),
- a **response mask** — `(isEos.cumsum − isEos) == 0`, keeping tokens up to and including the first eos (padding is excluded from all losses).

### 5.2 Group-relative advantage

For the `G` rewards in one group:

```
Âᵢ = (rᵢ − mean(r)) / (std(r) + 1e-8)
```

The **group mean is the baseline** — the empirical "expected reward for this prompt" that PPO's critic was trained to predict, now simply *measured*. The `std` is per-group whitening. Every token in a completion gets the same scalar advantage (outcome supervision). If all `G` rewards are equal (e.g. all format-only), the advantage is zero and that group teaches nothing — which is why reward *variance* (from the closeness term, §2) matters.

### 5.3 The reference and the KL penalty

The KL anchors the policy to **the SFT model** (where RL started — not the base), computed via the **adapter swap**: snapshot the current adapters, load the frozen SFT adapters, compute reference log-probs under `no_grad`, restore. One model in memory. The penalty uses the **k3 estimator** (low-variance, non-negative):

```
KL_t ≈ exp(Δ) − Δ − 1,     Δ = ref_logp − new_logp
```

### 5.4 The clipped update

For the frozen rollout, recompute `new_logp` **with gradients** and form the PPO-style clipped surrogate:

```
ratio      = exp(new_logp − old_logp)
policy     = − min(ratio·Â, clip(ratio, 1−ε, 1+ε)·Â)
loss       = masked_mean(policy + β·KL)            # over real response tokens
```

Only the LoRA adapters are optimized. On the first update `new_logp == old_logp` (ratio = 1); the clip bounds how far subsequent steps drift. The reference never changes.

### 5.5 Hyperparameters — the working recipe

| knob | value | note |
|---|---|---|
| prompts/iter `P` | 4 | |
| group size `G` | 8 (→4 when memory-bound) | bigger = less-noisy baseline |
| learning rate | 5e-5 | too low (1e-6) → no learning; too high → instability |
| KL coeff `β` | 0.02 | loose leash, lets the reasoning evolve |
| clip `ε` | 0.2 | standard |
| max new tokens | 256 | reasoning needs room — do **not** shrink it for memory |
| decoding | plain sampling | **no repetition penalty** (see JOURNEY) |

---

## 6. Evaluation & model selection

**The metric.** Per-sample solve rate — the fraction of generations that produce a valid expression equal to the target (`reward ≥ 1.0`). Measured on **100 held-out instances × 8 samples = 800 trials** (standard error ~1.7%).

**Why 800 trials.** A 5-instance × 8-sample eval (40 trials) is far too noisy to rank checkpoints — the same model varied by several points run to run. On the small set SFT and GRPO looked identical; only the 800-trial eval resolved the ~10-point GRPO gain.

**Checkpoint-and-select.** GRPO improves, peaks, then suffers an instability collapse late in training. So the policy is checkpointed every 10 iterations and the best is chosen by held-out solve rate — **not** the final step (which is often worse than the SFT start).

**Two numbers, two stories.** GRPO raised *per-sample* solve (50.9% → 60.4%) but *lowered* instance coverage (82 → 70 of 100 solved at least once). That's **policy sharpening** — RL concentrates probability on reliable solutions (higher per-attempt success) at the cost of the occasional lucky solve on borderline instances (lower breadth). Entropy reduction, made measurable.
