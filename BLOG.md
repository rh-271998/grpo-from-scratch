# I taught a tiny language model to reason — and watched it check its own work

The reasoning models everyone's talking about — the ones that "think" before answering — are trained with a technique called **RL with verifiable rewards**. The idea is simple and beautiful: for tasks where *checking* an answer is easy but *finding* it is hard (math, code, logic), you don't need a human or a learned judge — you reward the model against ground truth and let it figure out how to get there.

I wanted to understand that the only way I trust — by building the whole thing from scratch, on a small model, and watching it work. So I picked a tiny, honest version of the problem and built **GRPO** (the algorithm behind DeepSeek-R1) by hand: no RL libraries, every piece written in PyTorch.

Here's what happened — including the parts where it went wrong.

---

## The game

The task is **Countdown**: given four numbers and a target, combine them with `+ − × ÷` to hit the target.

> `[9, 1, 10, 8]`, target `82` → `9 × 1 × 10 − 8 = 82`

It's perfect for this, because *checking* an answer is a line of arithmetic, but *finding* one takes search. That gap is the whole reason you can skip a learned reward model and just reward the truth.

## Step 1: a model that can't play

Out of the box, the little 0.5B model can't do it — it rambles, wraps answers in LaTeX, invents numbers that weren't given. It has no idea it's supposed to produce a clean arithmetic expression.

## Step 2: teach it the *shape* of reasoning

First I fine-tuned it on examples that show the work step by step:

> `1 × 9 = 9. 10 + 14 = 24. 24 + 9 = 33.`

Crucially, I didn't just show clean solutions — I showed **failed attempts that get corrected**:

> `16 × 7 = 112 … 116 − 19 = 97. 97 is not 46. Let me try another way. 16 + 7 = 23 … 27 + 19 = 46.`

I wanted it to learn the *habit* of trying, checking, and retrying — not to memorize answers. After this, the model solves about **half** of held-out problems, and it reasons in clear steps.

## Step 3: reinforcement — make it better at getting it *right*

Now the actual RL. For each problem the model writes several attempts; I score each against the truth (exact answer = big reward, close = small reward, nonsense = zero) and nudge it toward the better ones. GRPO's clever trick is that it compares each attempt to the *average* of its siblings — so it needs no separate "critic" network to know what good looks like. It just asks: *was this attempt better than my other tries at the same problem?*

And something lovely emerged — the model started **checking its own work and backtracking**, for real, on problems it had never seen:

> `2 × 15 = 30. 2 + 14 = 16. 30 − 16 = 14. 14 is not 1. Let me try another way. 2 + 15 = 17. 17 − 2 = 15. 15 − 14 = 1.` ✓

It tries, notices it missed the target, and tries again until it lands. After RL, it solves **60%** of held-out problems — up from 50%.

## The part nobody tells you: it kept going wrong first

Getting there was mostly debugging my own mistakes, and they were instructive:

- **At first it solved problems but didn't actually reason** — the "thinking" was a copy-pasted placeholder and the answer was a direct guess. The fix wasn't the algorithm; it was teaching *varied* reasoning in the warm-up stage.
- **Then it lied about its own math** — "23 is not 26. This equals 26." Because every training example ended by asserting success, it learned to *always* claim success. Fixed by letting the reasoning end at its real value, and adding those backtracking examples.
- **Then I tried to stop it from repeating itself** with a standard decoding trick — and accuracy *dropped*. Countdown needs repeated numbers (`7, 7, 7`), and the trick punished exactly those. The right answer was to do nothing clever and let the reward handle it.
- **Then my scoreboard lied to me** — I was measuring on 5 problems, and the numbers bounced around randomly. Only when I tested on 100 held-out problems (800 attempts) did the real improvement become visible. Small evaluations invent and destroy results at random.

## The most honest finding

Here's the one I'm proudest of, because it's subtle. RL made the model **more reliable** (60% vs 50% per attempt) but on **fewer distinct problems** (70 vs 82 solved at least once). It traded *breadth* for *depth* — concentrating its confidence on solutions it could land dependably, at the cost of the occasional lucky solve on a hard problem. That's exactly what reward optimization does: it sharpens. Seeing it in my own numbers was the moment the theory became real.

---

## Why bother with the tiny version

Because I now understand — line by line, not abstractly — the machinery behind every reasoning model: how you turn "is this answer right?" into a training signal, how a group of samples replaces a value network, why the model learns to check and retry, and how easily a bad decoding trick or a too-small evaluation can fool you.

The task here was a number game. The machinery is the same one training the frontier reasoning models. That was the whole point: learn the thing that matters, on a problem small enough to see all of it — and be honest about where a 0.5B model's reasoning tops out.
