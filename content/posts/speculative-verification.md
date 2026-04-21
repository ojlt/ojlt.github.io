+++
title = "Speculative Verification"
date = "2026-04-21T12:00:00+01:00"
math = true

description = "How we aggregated nine shard-trained language-model heads at inference with a speculative-decoding-style verifier, defending against data poisoning at close to single-model speed."

tags = []
+++
# Speculative Verification

Most people would agree that a confidently wrong answer is worse than an uncertain correct one. One way to attack an LLM into being wrong is through data poisoning, where an adversary *poisons* the training data. Anthropic have a clear writeup of [how small the poisoned sample can be](https://www.anthropic.com/research/small-samples-poison). This blog post is inspired by work undertaken as part of my MSc project. Check out the [repo](https://github.com/msc-ai-mmmjo/tap) for more details.

## A defence by partitioning

One defence against poisoning is model aggregation. You train separate models on different shards of the data and pool their results at inference time. Doing this across entire models is clearly infeasible, so instead we take just the last three layers, a *head*, and fine-tune one on each shard. We train nine such heads on a single shared frozen trunk, each head seeing a disjoint ninth of the data. The architecture is OhLMo, a hydra built on OLMo-2-7B. The expensive bulk of the forward pass runs once in the trunk, and each head then produces its own logits.

![OhLMo architecture with speculative verification.](/images/tap-hydra.png)

*A shared trunk feeds the nine heads, which are composed at decode time by speculative verification. The diagram also shows an uncertainty head, which you can ignore for this post.*

## The inference problem

Having trained each head independently, aggregating their responses is not straightforward. With a large number of heads you could take a token-level vote, but that breaks down with only nine voters. Averaging all nine output distributions at every step is the obvious alternative, but it pays nine forward passes per token. We wanted the consensus of all nine heads at close to the cost of one.

## Speculative verification

Our solution is speculative verification, an aggregation framework similar in spirit to speculative decoding. The procedure runs as follows.

1. Sample a draft head, index $d$, uniformly from the nine heads.
2. Pass the prompt through the model. The draft head generates a short run of $\gamma$ tokens, with per-token probabilities $p_d$.
3. Pass the drafted tokens back through the model and read off a Product of Experts distribution from the remaining heads,

{{< math >}}
P(\cdot) = \frac{1}{Z} \exp\left( \beta \sum_{h \neq d} \log p_h(\cdot) \right),
{{< /math >}}

where $Z$ is the partition function and $\beta$ is an inverse temperature that sharpens or softens the result. This product is sharp where the heads agree, and it vanishes for any token that some head assigns near-zero probability.

4. Rejection-sample the drafted tokens against $P$. If a token is rejected, resample from the corrected distribution,

{{< math >}}
\tilde{x}_{t+i,\,\text{resample}} \sim \text{Norm}\left( \max\{0,\; P(\cdot \mid x_{\lt t+i}) - p_d(\cdot \mid x_{\lt t+i})\} \right).
{{< /math >}}

5. Append the verified chunk of $\gamma$ tokens and continue.

Because the draft proposes a whole chunk and the verifier checks it in a single pass, the verifier costs roughly $1/\gamma$ of a full averaged step rather than the full nine-head cost.

The same construction gives a per-token security guarantee. Fix a prompt and split the heads into an honest set $\mathcal{H}$ and a corrupted set $\mathcal{C}$, where a head is corrupted if its shard held enough poison to change its behaviour on that prompt. The accept-then-resample step is built so that the distribution actually emitted is exactly the Product of Experts distribution $P$, whichever head drafted (the derivation is in the project writeup). Written multiplicatively,

{{< math >}}
P(\cdot) = \frac{1}{Z} \prod_{h \neq d} p_h(\cdot)^{\beta}.
{{< /math >}}

Take any harmful token $x_{\text{harm}}$. Every factor satisfies $p_h(\cdot)^{\beta} \le 1$, so the whole product is bounded above by any single factor. If even one honest head $h \in \mathcal{H}$ sits among the verifiers and assigns the token small mass, $p_h(x_{\text{harm}}) < \epsilon$, then

{{< math >}}
P(x_{\text{harm}}) \;\le\; \frac{1}{Z}\, p_h(x_{\text{harm}})^{\beta} \;<\; \frac{1}{Z}\, \epsilon^{\beta}.
{{< /math >}}

A single confident, honest head is therefore enough to crush a harmful token's probability. The bound holds as long as the draft is checked against at least one clean head, that is $\mathcal{H}$ is non-empty among the verifiers, so the guarantee degrades gracefully as more heads are poisoned.

## Syncing the KV cache

Making this fast took real engineering on the KV cache. Standard caching assumes a single decode stream. Here the draft head and the verifier heads share a trunk but diverge in their own blocks, so each head needs its own cache, the caches have to stay in sync as tokens are accepted or resampled, and the verifier heads must consume the draft's trunk activations without re-running the trunk. We use separate per-head KV cache pointers, pointer syncing across heads, and a captured-hidden-state path for the shared trunk.

## Benchmarks

We benchmarked three decode strategies on a single NVIDIA L40 in bfloat16. Vanilla OLMo-7B, OhLMo with naive token averaging, and OhLMo with speculative verification. We report the best case, in which no draft tokens are rejected. There, speculative verification at $\gamma = 8$ recovers 86% of the single-tower throughput while still checking every token against a nine-head consensus, far above the naive-averaging floor of 55%.

| Configuration | Decode TPS |
|---|---|
| Baseline OLMo-7B | 23.3 |
| Hydra-7B (naive average, 9 heads) | 12.8 |
| Hydra-7B + spec. verification, $\gamma = 1$ | 12.6 |
| Hydra-7B + spec. verification, $\gamma = 2$ | 15.9 |
| Hydra-7B + spec. verification, $\gamma = 4$ | 18.5 |
| Hydra-7B + spec. verification, $\gamma = 8$ | 20.0 |

*Decode performance on an NVIDIA L40 (bfloat16), nine-head hydra, head depth 3. TPS is tokens per second. Best case with no resampling.*

The full system, training pipeline, and trained weights are open source at [github.com/msc-ai-mmmjo/tap](https://github.com/msc-ai-mmmjo/tap).
