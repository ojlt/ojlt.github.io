+++
title = "So what is PPO anyway?"
date = "2025-09-15T10:56:30+01:00"
math = true

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++
# So what is PPO anyway?
Proximal Policy Optimisation (PPO) is one of the most widely used algorithms in reinforcement learning. It is also a bit hard to understand. Here is its objective function in its full glory:
{{< math >}}
L^{CLIP}(\theta) = \mathbb{E}_{t}\left[\min\left(r_{t}(\theta)\hat{A}_{t}, \text{clip}(r_{t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{t}\right)\right]
{{< /math >}}

Make sense to you? If you're anything like me when you first read it, probably not. This blog post aims to present a clear explanation of the PPO algorithm, aiming for intuition but also rigour, as the elegance of PPO is best appreciated with some maths. This post assumes some basic familiarity with reinforcement learning jargon, and for a quick refresher, there is no better primer than this [blog](https://karpathy.github.io/2016/05/31/rl/) by Andrej Karpathy.

To understand PPO, you first need to understand Trust Region Policy Optimisation (TRPO), PPO's direct predecessor. To understand TRPO, you first need to understand the problems that on-policy RL faced.

## Pre-TRPO

Deep learning-based RL is similar to supervised learning in the sense that it is fundamentally a neural network optimisation problem. What makes on-policy RL difficult is that the policy (the neural network) that you are trying to *optimise* is also the policy that is *generating* the data you are optimising over. This means that if an optimisation step takes a policy to a barren part of the parameter space, the agent will generate its next set of experiences mostly based on this doomed policy. This is known as [catastrophic interference](https://en.wikipedia.org/wiki/Catastrophic_interference).

Catastrophic interference meant that on-policy algorithms often had extremely low learning rates, frequently taking millions of steps to converge to a satisfactory policy. In addition, the algorithms were extremely data-hungry, as there was just a single optimisation step per data-collection step.

## TRPO

TRPO was created in 2015 and aimed to address the problems outlined above. Before introducing TRPO, here is some notation.

- $\eta(\pi)$ is the expected return on a policy $\pi$. Explicitly, $\eta(\pi) = \sum_{\tau} P(\tau|\pi) \left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right]$, where $P(\tau|\pi) = p(s_0) \prod_{t=0}^{\infty} \pi(a_t|s_t)p(s_{t+1}|s_t, a_t)$ is the probability of a trajectory $\tau$ given the policy $\pi$.
- $Q^{\pi}(s,a)$ is the action-value function, which is the expected total reward the agent can get by taking action $a$ in state $s$ and following policy $\pi$ afterwards.
- $V^{\pi}(s)$ is the state-value function. This is the expected total reward the agent can get starting in state $s$ and following policy $\pi$ afterwards. It can also be represented in terms of $Q^{\pi}$: $V^{\pi}(s) = \sum_{a \in A} \pi(a|s) Q^{\pi}(s, a)$.
- $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$ is the advantage function. Intuitively, it is a measure of how good an action $a$ is in state $s$, compared to the *average* action the policy would take in state $s$.
- $p_{\pi}(s) = \mathbb{E}\_{\pi} \left[ \sum_{t=0}^\infty \gamma^t I_{S_t=s} \right]$ is the *discounted state visitation frequency*. It is a measure of how often we expect to encounter state $s$ following policy $\pi$, assigning heavier weights to earlier instances of $s$. Note that this is not actually a probability distribution.

Now consider two different policies $\pi$ and $\pi'$. To measure their relative performance, we could take the difference between their expected returns, denoted $\Delta_{\eta}(\pi,\pi')$:
{{< math >}}
\Delta_{\eta}(\pi,\pi') = \eta(\pi')-\eta(\pi)
{{< /math >}}

This form of $\Delta_{\eta}(\pi,\pi')$ is not the most useful, and I claim the following:
{{< math >}}
\begin{equation}
\Delta_{\eta}(\pi,\pi') = \mathbb{E}_{\pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^{\pi}(s_t, a_t)\right] = \eta(\pi')-\eta(\pi)
\end{equation}
{{< /math >}}

Here is a short derivation:
{{< math >}}
\begin{array}{rcl}
\mathbb{E}_{\pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^{\pi}(s_t, a_t)\right] & = & \mathbb{E}_{\pi'}\left[\sum_{t=0}^{\infty}\gamma^t(Q^{\pi}(s_t, a_t) - V^{\pi}(s_t))\right] \\
& = & \mathbb{E}_{\pi'} \left[ \sum_{t=0}^{\infty} \gamma^t (R_{t+1} + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)) \right] \\
& = & \mathbb{E}_{\pi'}\left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} + \sum_{t=0}^{\infty} (\gamma^{t+1}V^{\pi}(s_{t+1}) - \gamma^t V^{\pi}(s_t)) \right] \\
& = & \mathbb{E}_{\pi'}\left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} + \lim_{t \to \infty}\gamma^tV^{\pi}(s_{t+1}) - \gamma^0V^{\pi}(s_0)\right] \\
& = & \eta(\pi') - \mathbb{E}_{\pi'}[V^{\pi}(s_0)] \\
& = & \eta(\pi') - V^{\pi}(s_0)\mathbb{E}_{\pi'}[1] \\
& = & \eta(\pi')-\eta(\pi)
\end{array}
{{< /math >}}
Intuitively, this means that to compare $\pi$ and $\pi'$, we only need to consider how the advantage function of the original policy $\pi$ is weighted under the new policy $\pi'$.

Now that we've expressed $\Delta_{\eta}$ in a more workable form, let's see what happens when $\pi$ and $\pi'$ are very similar policies. In particular, we will assume that the *visitation frequencies* are similar:
$$
p_{\pi}(s) \approx p_{\pi'}(s)
$$
Using this assumption and equation (1), it follows that:
{{< math >}}
\begin{array}{rcl}
\Delta_{\eta}(\pi, \pi') & = & \mathbb{E}_{\pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^{\pi}(s_t, a_t)\right] \\
& = & \sum_{s}p_{\pi'}(s)\sum_a\pi'(a)A^{\pi}(s,a) \\
& \approx & \sum_{s}p_{\pi}(s)\sum_a\pi'(a)A^{\pi}(s,a)
\end{array}
{{< /math >}}

Remember that this $\Delta_{\eta}$ is a representation of how well $\pi'$ performs relative to $\pi$. The power of this approximation may not be obvious immediately and has to do with how we actually calculate $p_{\pi_0}$ for some generic policy $\pi_0$. Since we cannot take infinite environment steps, $p_{\pi_0}$ is always *approximated* from our environmental observations. Crucially, we cannot calculate $p_{\pi_0}$ until we have actually run policy $\pi_0$ in our environment. With this idea in mind, calculating
$$
 \sum_{s}p_{\pi'}(s)\sum_a\pi'(a)A^{\pi}(s,a) 
$$
would only be possible *after* taking a large number of environment steps under policy $\pi'$. This is extremely computationally expensive and is one of the key problems that TRPO set out to address. Put simply, we can now regard a parameter update as a pure optimisation step, optimising parameters over
{{< math >}}
\sum_{s}p_{\pi}(s)\sum_a\pi'(a)A^{\pi}(s,a)
{{< /math >}}
granted that the approximation $p_{\pi}(s) \approx p_{\pi'}(s)$ is well-founded. In the TRPO paper, a powerful bound was found involving the [$KL$ divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), which can roughly be understood as a measure of how similar two probability distributions are. In the TRPO paper, it was suggested that as long as the mean $KL$ divergence (denoted $\bar{D}_{KL}$) between two policies was small enough, the approximation between the visitation frequencies could yield strong results. In particular, if we denote the objective function as
{{< math >}}
L_{\pi}(\pi') = \sum_{s}p_{\pi}(s)\sum_a\pi'(a)A^{\pi}(s,a)
{{< /math >}}
then the TRPO algorithm is the following: Given some policy $\pi$, find the new policy by
{{< math >}}
\max_{\pi'}L_{\pi}(\pi') \quad \text{subject to} \quad \bar{D}_{KL}(\pi||\pi')<\delta
{{< /math >}}

And that is TRPO! Notice the following:
- The use of the trust region $\bar{D}_{KL}(\pi||\pi')<\delta$ ensures that policy updates never stray too far, which allows for much higher learning rates before running into the problem of catastrophic interference.
- One round of collected data yields multiple rounds of optimisation steps, meaning TRPO is much more sample-efficient than the on-policy algorithms that came before it.

Note that the form of the objective function above is not the form in which it is usually presented. It is typical to present it in terms of an expectation over the policy, which is not hard to derive.

{{< math >}}
\begin{array}{rcl}
L_{\pi}(\pi') & = & \sum_{s}p_{\pi}(s)\sum_a\pi'(a)A^{\pi}(s,a) \\
& = & \sum_{s}p_{\pi}(s)\sum_a\pi(a)\left( \frac{\pi'(a)}{\pi(a)}A^{\pi}(s,a) \right) \\
& = & \mathbb{E}_{\pi}\left[  \frac{\pi'(a)}{\pi(a)}A^{\pi}(s,a)
\right]
\end{array}
{{< /math >}}

## PPO

The original PPO paper introduced not one, but two new on-policy algorithms inspired by the success of the TRPO approach. Named clipped-objective PPO and adaptive KL PPO, the former is now the de facto "PPO" and is probably the most successful RL algorithm to date. Before learning how the clipped algorithm (which appeared at the start of this post) works, we introduce the adaptive KL approach.
### Adaptive KL PPO
TRPO is, at its core, an optimisation problem with a constraint dictated by the $KL$ divergence:
{{< math >}}
\max_{\pi'}\left(\mathbb{E}_{\pi}\left[  \frac{\pi'(a)}{\pi(a)}A^{\pi}(s,a)\right]\right) \quad \text{subject to} \quad \bar{D}_{KL}(\pi||\pi') < \delta
{{< /math >}}
The idea behind adaptive $KL$ PPO is to turn this constrained optimisation problem into a pure optimisation problem. Here is the proposed objective to maximise:

{{< math >}}
\max_{\pi'}\left(\mathbb{E}_{\pi}\left[  \frac{\pi'(a)}{\pi(a)}A^{\pi}(s,a)\right]  - \beta \bar{D}_{KL}(\pi||\pi') \right)
{{< /math >}}
Here, the parameter $\beta$ is a penalty coefficient that scales how much we care about the KL divergence constraint. A simple way to control this parameter (as suggested in the original PPO paper) is to set some target $KL$ divergence $d_{\text{tar}}$ and to update $\beta$ based on how the actual KL divergence $d = \bar{D}_{KL}(\pi||\pi')$ compares after each policy update.
- If $d < d_{\text{tar}} / 1.5$, set $\beta \leftarrow \beta / 2$ (the policy update is too conservative, so reduce the penalty).
- If $d > d_{\text{tar}} \times 1.5$, set $\beta \leftarrow \beta \times 2$ (the policy update is too aggressive, so increase the penalty).

While this approach seems pretty robust and is an intuitive follow-up to TRPO, it was found that it can be replaced by a far simpler algorithm. In fact, we can do away with the $KL$ divergence completely!

### Clipped Objective PPO
The idea behind clipped objective PPO is to focus on the probability ratio between the new and the old policies. If we denote the action and state at time $t$ by $a_t$ and $s_t$, respectively, then we denote the probability ratio $r_t(\theta)$ by

{{< math >}}
r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta, \text{old}}(a_t|s_t)}
{{< /math >}}
where we have changed notation from $\pi'$ to $\pi_{\theta}$ to match the original PPO paper. With this ratio in mind, the policy-improving objective can be written as
{{< math >}}
\mathbb{E}_t\left[ r_t(\theta)A_t \right]
{{< /math >}}
where $\mathbb{E}\_t$ is an expectation taken over all trajectories gathered from the old policy $\pi_{\theta,\text{old}}$. Now, if the advantage $A_t > 0$, we want to increase the probability of the corresponding action, meaning we want $r_t(\theta)>1$. Likewise, if $A_t < 0$, we want to decrease the probability, meaning we desire $r_t(\theta) < 1$. The ingenious trick in the clipped objective PPO algorithm was to enforce that $\pi_{\theta}$ and $\pi_{\theta,\text{old}}$ remain close not by using a KL divergence penalty, but by simply limiting how large or small the ratio $r_t(\theta)$ can be. Let's take a dive into the clipped PPO objective, as stated at the start of this post.

{{< math >}}
L^{CLIP}(\theta) = \mathbb{E}_{t}\left[\min\left(r_{t}(\theta)\hat{A}_{t}, \text{clip}(r_{t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{t}\right)\right]
{{< /math >}}

At this point, hopefully the terminology is clear, and the statement is readable. To understand what is happening inside the expectation, some graphs from the original PPO paper really help.
![foo](https://i.postimg.cc/CK2dVwDV/graph.png)
Let's analyse this case by case:
- When the advantage $A > 0$, the objective increases with $r_t(\theta)$. However, the clip function prevents the ratio from exceeding $1 + \epsilon$, which stops the policy from changing too aggressively.
- When the advantage $A < 0$, the objective is driven by making $r_t(\theta)$ smaller. The clip function prevents the ratio from going below $1 - \epsilon$, again limiting how large the policy update can be.

Simply put, we encourage $\pi_{\theta}$ and $\pi_{\theta, \text{old}}$ to be close, by simply clipping how big the ratio between the two can get! This offers a much simpler and faster implementation that $KL$ divergence based approaches.

Two years later, OpenAI used PPO to [defeat pro players at Dota 2](https://www.youtube.com/watch?v=UZHTNBMAfAA), and a slightly altered version on PPO, GRPO is being used to [train LLM's to reason](https://arxiv.org/pdf/2402.03300). While I haven't gone into all of the details (there is more to say on the advantage estimation, for example), I hope this post has given you some insight into the development of PPO!