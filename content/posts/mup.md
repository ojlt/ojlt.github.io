+++
title = "The Maximal Update Parametrisation"
date = "2026-05-30T12:00:00+01:00"
math = true

description = "Why μP keeps a network's training dynamics width-invariant, so the hyperparameters you tune on a small model transfer to a large one."

tags = []
+++
# The Maximal Update Parametrisation

Training a frontier model is expensive enough that you essentially get one shot at the hyperparameters. Sweeping the learning rate over a billion-parameter network is a non-starter, yet that one number is the difference between a state-of-the-art model and a divergent one. So here is a tempting idea. Tune the small model you *can* afford to sweep, and carry those choices over to the big one.

The catch is that the best learning rate usually drifts as a model grows, so what is optimal at 100M is wrong at 100B. The **Maximal Update Parametrisation (μP)** is the fix. It is a recipe for *how to scale* a network's initialisation, learning rates, and multipliers with width so that the optimal hyperparameters stop drifting and instead sit at the same value at every size. Tune once on a small proxy, transfer for free. Its authors call this **μTransfer**.

This post builds up *why* it works, starting from a single fact about large sums. I have aimed for intuition but kept the maths honest, because the elegance of μP is hard to appreciate otherwise. It also covers what μP gives you beyond cheaper tuning, which is predictable scaling.

## TL;DR

The whole idea in three points, before we derive it.

1. Training is the forward pass, backward pass, and parameter update, each a tensor contraction over the width. What we track are the typical per-coordinate sizes of the activations, gradients, and updates.
2. We want each of those to be $\Theta(1)$ and width-invariant, so the whole training trajectory is width-invariant.
3. **Correlation decides the regime**, $\Theta(\sqrt{n})$ when the operands are independent and $\Theta(n)$ when they are correlated. Initialisation lives in the independent forward pass at the start of training. The correlated updates live in the $\Theta(n)$ regime, so init and learning rate must scale differently. Choosing the init, learning rate, and multipliers so every per-coordinate magnitude comes out $\Theta(1)$ *is* the parametrisation.

## Scaling training dynamics

What exactly do we need to hold steady as the model grows? The answer is the *training dynamics*, the three operations training repeats at every step.

- Forward pass, $\vec{h} = \phi(\mat{W}\vec{x})$.
- Backward pass, $\nabla_{\mat{W}}\loss$.
- Parameter update, $\mat{W} \leftarrow \mat{W} - \eta\, \nabla_{\mat{W}}\loss$.

We take a local view. Pick a single activation coordinate or weight, look at its typical magnitude, and demand that this stays put as the width grows. "Width" here means the hidden size $n$ (or $d_\text{model}$ in a Transformer), measured against a fixed base model through the width multiplier $\wmult \defeq d / d_\text{base}$, so that the base model is $\wmult = 1$.

If those typical magnitudes never move with width, every hyperparameter plays the same role at every scale, and its optimum transfers from the small model to the large one. That is exactly what μP buys us, under the name μTransfer.

## Intuition and motivating examples

Every operation in training is a *tensor contraction*, a sum over an index running across the width $n$. The matrix-vector products of the forward and backward passes are contractions, and so are the outer-product updates. The size of such a sum is set by the typical per-coordinate size of the vectors entering it, so that per-coordinate size is the thing we control.

The key insight behind μTransfer is to analyse these sums with two classical results, the law of large numbers and the central limit theorem. They turn out to be all we need.

### The law of large numbers and the central limit theorem

Let $x_1, \dots, x_n$ be independent samples of a random variable $X$. The law of large numbers says their average converges to the mean,

{{< math >}}
\frac{1}{n}\sum_{i=1}^{n} x_i \to \E[X], \tag{*}
{{< /math >}}

while the central limit theorem says the fluctuation about the mean, rescaled by $\sqrt{n}$, converges to a Gaussian,

{{< math >}}
\frac{1}{\sqrt{n}}\sum_{i=1}^{n} \bigl(x_i - \E[X]\bigr) \to \Normal\!\bigl(0,\, \sigma^2(X)\bigr), \tag{**}
{{< /math >}}

where $\sigma^2(X)$ is the variance of $X$.

What we actually care about is the size of the *unscaled* sum $S_n = \sum_{i=1}^{n} x_i$. Multiplying $(*)$ by $n$ gives $S_n \approx n\,\E[X]$, which pins the size whenever the mean is nonzero. When $\E[X] = 0$ that estimate only says $S_n$ grows slower than $n$. The finer result $(**)$, multiplied by $\sqrt{n}$, fills in the rest. So there are two regimes.

- **Nonzero mean** ($\E[X] \neq 0$). The sum is dominated by its mean, $S_n \approx n\,\E[X]$, of typical size $\Theta(n)$.
- **Zero mean** ($\E[X] = 0$). The mean contributes nothing and the surviving fluctuation gives $S_n \approx \sqrt{n}\,\Normal(0, \sigma^2)$, of typical size $\Theta(\sqrt{n})$.[^cltsize]

This contrast is the engine of everything that follows. We get $\Theta(n)$ when a coherent mean accumulates, and $\Theta(\sqrt{n})$ when zero-mean terms partially cancel.

It helps to make "typical magnitude" precise for a vector. We say $\vec{v} \in \R^{n}$ has $\Theta(n^a)$-sized coordinates when

{{< math >}}
\frac{\norm{\vec{v}}_2^2}{n} = \Theta(n^{2a}),
{{< /math >}}

i.e. a representative entry has typical size $\Theta(n^a)$. Equivalently $\norm{\vec{v}}_2 = \Theta(n^{a + 1/2})$, so the quantity we track is the root-mean-square coordinate, not any single entry (which may fluctuate). Our width-invariance requirement is then simply that activations carry $\Theta(1)$-sized coordinates at every width.

[^cltsize]: The $\approx$ here is shorthand for convergence in distribution, $S_n / \sqrt{n} \to \Normal(0, \sigma^2)$. The sum does not approach a fixed value. It fluctuates at scale $\sqrt{n}$, exceeding any fixed multiple of $\sqrt{n}$ with probability bounded away from zero as $n \to \infty$. It is this scale, not a limiting value, that we call the typical size.

### Two motivating examples

A network is a chain of matrix-vector products, and the matrices come in two kinds. At initialisation a weight is a random matrix. During training it picks up increments built from the activations passing through it. The next two examples are the *same* hidden weight, before and after a single step, and they land in the two regimes above. That is exactly why they must be scaled differently.

**Example (a weight at initialisation).** Let $\mat{W} \in \R^{n \times n}$ have independent zero-mean entries of variance $\sigma_W^2$, acting on an input $\vec{x}$ with $\Theta(1)$-sized coordinates, independent of $\mat{W}$. Each output coordinate

{{< math >}}
y_i = \sum_{j=1}^{n} W_{ij} x_j
{{< /math >}}

sums $n$ zero-mean independent terms, so it is in the central limit regime. So $y_i$ has typical size $\Theta(\sqrt{n})$. Equivalently $\Var[y_i] = n\,\sigma_W^2\,\sigma_x^2$, so choosing $\sigma_W^2 = 1/n$ brings it back to $\Theta(1)$, which is just the Glorot/He initialisation.

**Example (the same weight after one step).** After one gradient step the weight has an increment. Writing $\vec{\delta} = \partial \loss / \partial \vec{y}$ for the backpropagated error, the gradient is the outer product $\vec{\delta}\,\vec{x}\T$, so

{{< math >}}
\mat{W}' = \mat{W} - \eta\,\vec{\delta}\,\vec{x}\T.
{{< /math >}}

Feed in the same input again,

{{< math >}}
\mat{W}'\vec{x} = \mat{W}\vec{x} - \eta\,\vec{\delta}\,(\vec{x}\T\vec{x}),
{{< /math >}}

and now the inner product $\vec{x}\T\vec{x} = \sum_{j=1}^{n} x_j^2$ sums $n$ strictly positive, hence nonzero-mean, terms. By the law of large numbers it is $\Theta(n)$, so with $\eta$ and $\vec{\delta}$ of order one the increment shoves each activation coordinate by an amount of order $\Theta(n)$. The update is built from the very vector it then multiplies, so $\vec{x}$ paired against itself cannot cancel.

The two examples differ in just one thing, the mean of a typical term. At init the terms $W_{ij} x_j$ are zero-mean (since $\mat{W}$ is centred and independent of $\vec{x}$), and the sum cancels down to $\Theta(\sqrt{n})$. After the step the terms $x_j^2$ have positive mean $\E[x_j^2] = \Var(x_j) > 0$, and the sum piles up to $\Theta(n)$. **Correlation is the switch between the two regimes**, and at a fixed entry scale they sit a factor of $\sqrt{n}$ apart. A weight's initialisation and its updates therefore cannot be scaled the same way. That asymmetry is the seed μP grows from.

The same demand applies to the *whole* of training. Every operation, forward, backward, and update, should preserve the typical size of what it produces, at init and at every step after. Working out the scaling that achieves this, weight by weight, is the job of the rest of this post.

## Gaussian matrices and tensor products

The two examples are instances of two general families of matrix, and the scaling of any operation is fixed by which family its matrices belong to. Here are all four cases we need, with the two extras contributed by adaptive optimisers and the readout layer.

- **Tensor-product matrices**, an SGD weight update.
- **Nonlinear tensor-product matrices**, an Adam update.
- **Gaussian matrices**, a weight at initialisation.
- **Vectors**, the readout layer.

Throughout, a vector $\vec{v} \in \R^{n}$ with roughly independent $\Theta(1)$-sized coordinates is summarised by its *coordinate distribution* $Z_v$, the law of a representative entry. We take a matrix $\mat{A}$ with $\Theta(1)$-sized entries acting on such a $\vec{v}$ and ask how big a coordinate of $\mat{A}\vec{v}$ is.

### Tensor-product matrices

A single SGD step adds an outer product of two $\Theta(1)$-coordinate vectors,

{{< math >}}
\mat{A} = \vec{u}\,\vec{v}\T, \qquad A_{\alpha\beta} = u_\alpha v_\beta = \Theta(1).
{{< /math >}}

Acting on $\vec{x}$ with $\Theta(1)$-sized coordinates,

{{< math >}}
(\mat{A}\vec{x})_\alpha = u_\alpha \sum_{\beta=1}^{n} v_\beta x_\beta = u_\alpha\,(\vec{v}\T\vec{x}).
{{< /math >}}

The inner product is governed by the law of large numbers,

{{< math >}}
\frac{1}{n}\,\vec{v}\T\vec{x} = \frac{1}{n}\sum_{\beta=1}^{n} v_\beta x_\beta \;\longrightarrow\; \E[Z_v Z_x]. \tag{†}
{{< /math >}}

The decisive feature, again, is correlation. The update is assembled from the same activations that make up $\vec{x}$, so $\vec{v}$ and $\vec{x}$ are correlated and the expectation does *not* factorise as $\E[Z_v Z_x] = \E[Z_v]\,\E[Z_x]$. Even for a centred input we cannot argue $\E[Z_v Z_x]$ vanishes, so the central limit cancellation is unavailable. The products accumulate by the law of large numbers instead. By (†), $\vec{v}\T\vec{x}$ is $\Theta(n)$, so $\mat{A}\vec{x}$ has $\Theta(n)$-sized coordinates. (Only if $\vec{v}$ and $\vec{x}$ were independent would the expectation factorise and, for centred input, vanish.)

So the raw outer product drives $\mat{A}\vec{x}$ to $\Theta(n)$. Since that grows linearly in $n$, dividing the matrix by $n$ cancels it. The scaled update

{{< math >}}
\mat{A} = \frac{1}{n}\,\vec{u}\,\vec{v}\T
{{< /math >}}

gives $(\mat{A}\vec{x})_\alpha \to u_\alpha\,\E[Z_v Z_x]$, of $\Theta(1)$ size, with $Z_{Ax} = Z_u\,\E[Z_v Z_x]$. **A correctly scaled tensor-product matrix has $\Theta(1/n)$ entries.** A sum of $k$ such outer products, $\mat{A} = \tfrac{1}{n}\sum_{i=1}^{k} \vec{u}_i \vec{v}_i\T$, gives $Z_{Ax} = \sum_{i=1}^{k} Z_{u_i}\,\E[Z_{v_i} Z_x]$.

### Nonlinear tensor-product matrices

Adam normalises the gradient coordinatewise before applying it, so its increment is no longer a clean outer product. With $k$ accumulated gradients carrying factors $\vec{u}^1,\dots,\vec{u}^k$ and $\vec{v}^1,\dots,\vec{v}^k$, a typical entry is

{{< math >}}
A_{\alpha\beta} = \psi\bigl(u_\alpha^1,\dots,u_\alpha^k,\; v_\beta^1,\dots,v_\beta^k\bigr),
{{< /math >}}

for a fixed $\psi : \R^{2k} \to \R$. For Adam this is the first moment over the square root of the second,

{{< math >}}
A_{\alpha\beta} = \frac{\sum_i \gamma_i\, u_\alpha^i v_\beta^i}{\sqrt{\,\sum_i \omega_i\,(u_\alpha^i v_\beta^i)^2\,}},
{{< /math >}}

with $\gamma_i, \omega_i$ the moving-average weights. We call such an $\mat{A}$ a *nonlinear tensor-product matrix*. The linear case is just $\bar\psi = \sum_i u_\alpha^i v_\beta^i$. Adam's normalisation leaves the entries at size $\Theta(1)$ (written $\bar\psi$ below), so the raw action is again $\Theta(n)$ by the law of large numbers, the factors $\vec{v}^i$ correlating with $\vec{x}$ just as before. Dividing by $n$, that is taking $\psi = n^{-1}\bar\psi$, restores $\Theta(1)$,

{{< math >}}
(\mat{A}\vec{x})_\alpha = \frac{1}{n}\sum_{\beta=1}^{n} \bar\psi\bigl(u_\alpha^1,\dots,\,v_\beta^1,\dots\bigr)\, x_\beta \;\longrightarrow\; \E\bigl[\bar\psi\bigl(u_\alpha^1,\dots,Z_{v^1},\dots\bigr)\,Z_x\bigr] \eqdef \Psi(u_\alpha^1,\dots,u_\alpha^k),
{{< /math >}}

with $o(1)$ error. So $\mat{A}\vec{x}$ has $\Theta(1)$-sized coordinates, $Z_{Ax} = \Psi(Z_{u^1},\dots,Z_{u^k})$. The unnormalised Adam update has $\Theta(1)$ entries, a factor of $n$ too big, so it wants the same $\Theta(1/n)$ scaling as the linear case.

### Gaussian matrices

At initialisation a hidden weight is a Gaussian matrix $\mat{A} \in \R^{n \times n}$ with independent zero-mean entries. For $\vec{x}$ independent of $\mat{A}$ with $\Theta(1)$-variance entries,

{{< math >}}
(\mat{A}\vec{x})_\alpha = \sum_{\beta=1}^{n} A_{\alpha\beta}\, x_\beta
{{< /math >}}

sums $n$ zero-mean independent terms, so it is $\Theta(\sqrt{n})$ by the central limit theorem, a full factor of $\sqrt{n}$ smaller than a tensor-product matrix with the same entry size. To reach $\Theta(1)$ we need variance $\Theta(1/n)$, that is entries of size $\Theta(1/\sqrt{n})$, the $\sigma_W^2 = 1/n$ from the first example. (Remarkably, this $\Theta(\sqrt{n})$ scaling survives even when $\vec{x}$ correlates with $\mat{A}$, as it does once $\mat{A}$ has been used earlier. That is the subtle point the readout case turns on, below.)

### The readout vector

This is the trickiest of the four, and the one that earns μP its careful treatment. The readout maps the width-$n$ hidden state to $V$ logits, one per vocabulary token. Each logit contracts one row $\mat{A} \in \R^{1 \times n}$ of the readout matrix with the hidden state, $\sum_{\beta=1}^{n} A_\beta v_\beta$, a sum over the width, and since $V$ is fixed, one row tells the whole story.

At initialisation $\mat{A}$ is independent of $\vec{v}$, so the contraction is $\Theta(\sqrt{n})$ by the central limit theorem. So far this is *exactly* the Gaussian matrix case, and you would be forgiven for concluding we just scale it the same way, a $1/\sqrt{n}$ init, and move on.

Not so fast. Recall the subtle claim from the Gaussian case. An $n \times n$ weight holds at $\Theta(\sqrt{n})$ *even once its input correlates with it*. The readout is a contraction of the very same form, so surely the same protection applies? It does not, and seeing why is the whole point. Once training correlates $\mat{A}$ with its input the readout grows to $\Theta(n)$, the tensor-product regime, so it needs entries of size $\Theta(1/n)$, not $\Theta(1/\sqrt{n})$. A full matrix keeps the $\sqrt{n}$ protection. A single row loses it. Why?

Split any trained weight into its init and its updates, $\mat{W} = \mat{W}_0 + \Delta\mat{W}$. The update is a correlated tensor-product matrix for every weight alike, $\Theta(n)$ pulled back to $\Theta(1)$ by the learning rate. The difference is the *initial* piece $\mat{W}_0\vec{x}$. By training time $\vec{x}$ depends on $\mat{W}_0$, and the question is whether that dependence survives the sum.

The forward pass contracts over the *input* dimension, and from the forward pass alone a hidden weight ($n \times n$) and the readout ($V \times n$) are indistinguishable. Both contract a width-$n$ input,

{{< math >}}
\text{(hidden)}\quad y_\alpha = \sum_{\beta=1}^{n} (W_0)_{\alpha\beta}\, x_\beta, \qquad \text{(readout)}\quad \mathrm{logit}_v = \sum_{i=1}^{n} (W)_{vi}\, z_i,
{{< /math >}}

The dependence of the input on the weight is created in the *backward* pass, which contracts over the *output* dimension,

{{< math >}}
\text{(hidden)}\quad \frac{\partial L}{\partial x_\beta} = \sum_{\gamma=1}^{n} (W_0)_{\gamma\beta}\,\delta_\gamma, \qquad \text{(readout)}\quad \frac{\partial L}{\partial z_i} = \sum_{v=1}^{V} (W)_{vi}\,\frac{\partial L}{\partial \mathrm{logit}_v}.
{{< /math >}}

The gradient into an input coordinate sums over the outputs, so the single entry a given output uses is one row out of the output dimension. Its correlation with its input partner is of order $1/(\text{output dim})$, and the covariance built up over the input-dim-many forward terms scales as

{{< math >}}
\frac{\text{input dimension}}{\text{output dimension}}.
{{< /math >}}

For a hidden weight both dimensions are the width, so the ratio is $\Theta(1)$ and the dilution keeps pace with the sum. The covariance stays below the $\Theta(\sqrt{n})$ central-limit fluctuation, and $\mat{W}_0\vec{x}$ stays $\Theta(\sqrt{n})$. For the readout the output dimension is the fixed $V$, so the ratio $n/V$ *grows* with width. A finite output cannot dilute a growing input sum. The covariance overruns the fluctuation, and the readout's initial piece becomes $\Theta(n)$. So the readout is *not* the matrix case after all. The fixed output dimension is the whole difference. That is what "a single row has no spare randomness" means. The dilution is counted along the output dimension, and that dimension is finite.

The backward pass is the mirror image. Now $g_x = \mat{W}\T g_y$ contracts over the output and dilutes by the input, so its covariance scales as $\text{output}/\text{input}$. A hidden weight is balanced both ways, which is why its backward pass needs no rule. The readout is imbalanced. Its forward ratio $n/V$ grows while its backward ratio $V/n$ vanishes, so the gradient leaving it is the clean side and all the trouble is in the forward. The embedding is the opposite mirror, a finite *input* contracted in the forward pass, accumulating no width-covariance and needing no width scaling at all. (This is the classification at the heart of the Tensor Programs framework. Sort each weight by which dimensions are infinite and which finite. Hidden weights are infinite in both, output weights infinite in but finite out, input weights the reverse.)

A finite-width caveat is worth a line. The readout's $n/V$ only overruns its $\Theta(\sqrt{n})$ fluctuation once $n \gtrsim V^2$, so in the realistic regime $n < V$ the readout is still fluctuation-dominated and the asymptotic rule is an extrapolation. This is the alignment question studied by Everett et al.

This also explains an asymmetry you will meet in the Transformer table below. The readout takes a *multiplier* of $1/n$ but keeps a constant learning rate, whereas a hidden weight scales its learning rate as $1/n$. A hidden weight must keep *both* its forward ($\sim\sqrt{n}$) and its update ($\sim n$) at $\Theta(1)$, two demands that force two separate knobs, a $1/\sqrt{n}$ init and a $1/n$ learning rate. The readout is allowed to vanish at init, so it has no forward constraint and only its update need be controlled. A single multiplier of $1/n$ does the whole job and leaves the learning rate alone.

| Matrix $\mat{A}$, entries $\Theta(1)$ | Arises as | Size of $\mat{A}\vec{v}$ |
|---|---|---|
| Gaussian, $n \times n$ | initial weights | $\Theta(\sqrt{n})$ |
| Tensor product, $n \times n$ | SGD update | $\Theta(n)$ |
| Nonlinear tensor product, $n \times n$ | Adam update | $\Theta(n)$ |
| Vector, $1 \times n$ | readout layer | $\Theta(n)$ |

*Typical size of a coordinate of $\mat{A}\vec{v}$, for $\vec{v}$ of $\Theta(1)$-sized coordinates and $\mat{A}$ of $\Theta(1)$-sized entries, correlated as they are during training. A Gaussian matrix works through the central limit theorem ($\Theta(\sqrt{n})$), while a tensor-product matrix and the readout vector work through the law of large numbers ($\Theta(n)$). The readout is $\Theta(\sqrt{n})$ only at init, before its input correlates with it.*

**Remark (the backward pass and the loss).** The backward pass needs no rule of its own. Propagating a gradient through a weight multiplies by $\mat{W}\T$, a contraction of the same form as the forward pass and update, so the same initialisation and learning rate hold it in place. The scale of the loss, and so of the gradient $\vec{\delta}$ that seeds the backward pass, is set by the readout multiplier, which keeps the output at $\Theta(1)$. That is what lets us treat $\vec{\delta}$ as order one in the second example.

## Parametrising a Transformer

Now we just apply those rules to a decoder-only Transformer, reading each parameter's scaling off the case it falls under. The standard derivations, from Yang et al. and the Cerebras guide, prescribe the same parametrisation. Where they differ it is only in notation or a modelling choice, which I flag in passing. We swap the absolute width $n$ for the multiplier $\wmult = d_\text{model}/d_{\text{model},0}$ from earlier. Since $n = \wmult\, n_0$ and the constant $n_0$ does not affect the order, a $1/n$ scaling simply becomes $1/\wmult$. The result is the table below.

- **Hidden weights** (the attention projections $\mat{W}_q, \mat{W}_k, \mat{W}_v, \mat{W}_o$ and the two feed-forward matrices). Matrix-like, and governed by two cases at once. At init they are Gaussian, so an init variance of $\Theta(1/\wmult)$ (entry size $\Theta(1/\sqrt{\wmult})$) holds the forward pass at $\Theta(1)$. Under training they are correlated tensor-product increments, so their entries scale as $\Theta(1/\wmult)$, which for Adam is a learning rate of $\Theta(1/\wmult)$. All three sources agree, and this pairing is the heart of μP.

- **Input embeddings.** With $\mat{W}_\text{emb} \in \R^{d_\text{model} \times V}$ and a one-hot token $\vec{e}_t$,

{{< math >}}
\vec{h} = \mat{W}_\text{emb}\,\vec{e}_t, \qquad h_i = (\mat{W}_\text{emb})_{i,t}.
{{< /math >}}

  The contraction is over the vocabulary $V$, which is fixed in width, so each coordinate is a single weight entry with no width-sum to grow. Neither limit theorem applies. A width-independent init gives $\Theta(1)$ activations for free, with a constant learning rate and a tunable multiplier $\alpha_\text{emb}$.

- **Unembedding.** With $\mat{W}_\text{unemb} \in \R^{V \times d_\text{model}}$ and final hidden state $\vec{z} \in \R^{d_\text{model}}$, a raw logit is the width-contraction

{{< math >}}
\mathrm{logit}_v = \sum_{i=1}^{d_\text{model}} (W_\text{unemb})_{vi}\, z_i,
{{< /math >}}

  the readout vector from above. At init $\vec{z}$ and $\mat{W}_\text{unemb}$ are independent, so the sum is $\Theta(\sqrt{\wmult})$. Once training correlates them it grows to $\Theta(\wmult)$. The multiplier

{{< math >}}
\mathrm{logits} = \frac{\alpha_\text{out}}{\wmult}\,\mat{W}_\text{unemb}\,\vec{z}
{{< /math >}}

  is sized for the trained regime. Its $1/\wmult$ brings the trained logits to $\Theta(1)$, and at init the logits are then a harmless $\Theta(1/\sqrt{\wmult})$. (The init-only $1/\sqrt{\wmult}$ will not do, since it would let the trained logits blow up as $\sqrt{\wmult}$. The Cerebras guide makes this point.)

- **Attention logit.** The contraction $\vec{q}\T\vec{k}$ over the head dimension $d_\text{head}$. Independent at init, hence the usual $1/\sqrt{d_\text{head}}$, but correlated under training, so it grows as $d_\text{head}$ and μP uses $1/d_\text{head}$ throughout. Yang et al. and the Cerebras guide adopt this. An alternative is to fix $d_\text{head}$ and grow width through the number of heads instead, in which case the choice does not affect transfer.

- **Biases and layernorm.** Vector-like, with a single width dimension and no contraction over it, so init (the usual $1$ and $0$), learning rate, and multiplier are all constant in width. No one scales them.

The accounts differ only in presentation otherwise. Yang et al. write the rules via the input/output dimensions rather than $\wmult$, in three equivalent tables related by shuffling scale between init, multiplier, and learning rate. They also handle SGD, where the hidden learning rate splits differently across layers. The Adam form here is the one the Cerebras guide uses directly. One shared refinement is to initialise the unembedding and the query projection at zero, which removes a Gaussian-process mismatch between proxy and target at init.

| Parameter | Init. variance | Multiplier | Adam LR | Origin |
|---|---|---|---|---|
| Hidden (QKV, O, MLP) | $\Theta(1/\wmult)$ | $1$ | $\Theta(1/\wmult)$ | init (Gaussian), update (tensor product) |
| Embeddings (input) | $\Theta(1)$ | $\alpha_\text{emb}$ | $\Theta(1)$ | finite contraction |
| Unembedding (output) | $\Theta(1)$ | $\alpha_\text{out}/\wmult$ | $\Theta(1)$ | readout / vector case |
| Attention logit | n/a | $1/d_\text{head}$ | n/a | tensor product |
| Biases, LayerNorm | $\Theta(1)$ | $1$ | $\Theta(1)$ | finite contraction |

*μP for a decoder-only Transformer trained with Adam, as scalings in the width multiplier $\wmult$. Here $\Theta(1)$ means no change with width, $\alpha_\text{emb}$ and $\alpha_\text{out}$ are tunable width-independent multipliers, and $d_\text{head}$ is the per-head dimension. The origin column points to the case each rule comes from.*

That is the whole recipe. Read each weight's scaling off the case it falls into, set the init, learning rate, and multipliers so that every per-coordinate magnitude comes out $\Theta(1)$, and the training dynamics, and with them the optimal hyperparameters, stop moving with width.

## μTransfer and predictability

In μP the optimal hyperparameters are stable across width. Parametrise the target model in μP, sweep the hyperparameters on a small proxy you can afford to train, and use the same values on the full-sized model. This is μTransfer. The figure shows it. Under standard practice the optimal learning rate drifts as the model widens, so the value found on a small model is wrong on a large one. Under μP the curves line up and the optimum stays in the same place.

![μP keeps the optimal learning rate fixed across width, while standard practice lets it drift.](/images/mup-lr-transfer.png)

The original paper reports the savings directly. Transferring from a 13M-parameter proxy beat the published BERT-large (350M) results. Transferring from a 40M proxy matched GPT-3 6.7B, with tuning costing about 7% of a single pretraining run.

The same stability makes training behaviour predictable. Because the per-coordinate magnitudes do not change with width, a small model and a large one follow the same dynamics, so the loss of a large model can be projected from smaller runs. The Cerebras-GPT models show this. Under standard practice their loss sits above and below the scaling-law trend from one size to the next. Under μP they fall on a smooth line that extrapolates cleanly.

![Trained with μP, models track a smooth scaling-law trend, while standard practice scatters around it.](/images/mup-cerebras-predictability.png)

## References

- Greg Yang et al. (2022), [*Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*](https://arxiv.org/abs/2203.03466). The μP and μTransfer paper.
- Cerebras & EleutherAI, [*The Practitioner's Guide to the Maximal Update Parameterization*](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization) ([EleutherAI mirror](https://blog.eleuther.ai/mutransfer/)).
- Katie Everett et al. (2024), [*Scaling Exponents Across Parameterizations and Optimizers*](https://arxiv.org/abs/2407.05872). The finite-width alignment study.
- Xavier Glorot & Yoshua Bengio (2010), [*Understanding the difficulty of training deep feedforward neural networks*](https://proceedings.mlr.press/v9/glorot10a.html), and Kaiming He et al. (2015), [*Delving Deep into Rectifiers*](https://arxiv.org/abs/1502.01852). The classical initialisations recovered above.
- Yao & Wang. *Reference to add.*
