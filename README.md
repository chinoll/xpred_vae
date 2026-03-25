# 为什么 VAE latent + JiT 风格 `x-pred` 会让 DiT 用更少训练步数达到相同效果？
在 **VAE latent** 上训练 DiT 时，如果把更常见的 `eps-pred` 或 `v-pred` 换成 **JiT 风格的 `x-pred` 参数化**，模型往往可以用显著更少的训练步数达到相同的生成质量。

---

## 1. 为什么真正要比较的是 VAE latent 上的 `x` vs `eps/v`

在 VAE latent diffusion 里，网络通常可以直接输出三种对象之一：

- `eps-pred`：直接输出噪声 \(\epsilon\)
- `v-pred`：直接输出 \(v\)
- `x-pred`：直接输出 clean latent \(x\)

很多讨论喜欢只比较 `x-pred` 和 `eps-pred`。但如果我们想解释 **VAE latent + JiT 风格 `x-pred`** 为什么会显著加速 DiT，真正应该比较的是：

> `x-pred` 对比所有仍然显式含有噪声分量的输出空间，也就是 `eps-pred` 和 `v-pred`。

因为 `v-pred` 虽然比纯噪声预测更接近数据，但它只要还含有非零噪声分量，本质上就仍然不是纯净的数据流形回归问题。

所以本文的主张不是：

> `x-pred` 只是比 `eps-pred` 好一点，

而是：

> `x-pred` 和 `eps-pred` / `v-pred` 在网络输出几何上根本属于两类问题。

---

## 2. 一个最基本的建模视角：先进入 VAE latent，再决定网络直接输出什么

设 VAE 编码器为

$$
E_{\text{VAE}}(\cdot),
$$

输入图像为 \(I\)，则 clean latent 为

$$
x = E_{\text{VAE}}(I) \in \mathbb{R}^D.
$$

接下来，我们不是在 pixel space 上定义扩散，而是在这个 VAE latent 上定义扩散过程：

$$
z_t = \alpha_t x + \sigma_t \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I_D).
$$

在这个 VAE latent diffusion transformer 中，网络可以直接输出下面三种对象之一：

### `x-pred`

$$
f_\theta(z_t, t) \approx x
$$

### `eps-pred`

$$
f_\theta(z_t, t) \approx \epsilon
$$

### `v-pred`

把 `v` 写成更一般的形式：

$$
v_t = \beta_t x + \gamma_t \epsilon.
$$

于是

$$
f_\theta(z_t, t) \approx v_t.
$$

从“网络直接输出什么”这个角度看，这三类形式可以统一写成：

$$
y_t = a_t x + b_t \epsilon.
$$

其中：

- `x-pred`：\(a_t=1, b_t=0\)
- `eps-pred`：\(a_t=0, b_t=1\)
- `v-pred`：\(a_t=\beta_t, b_t=\gamma_t\)

这说明：

> `x-pred` 是唯一一个完全不显式保留噪声分量的**网络输出空间**。

这件事在理论上非常关键。

---

## 2.5 你的训练形式：VAE latent `x-pred`（JiT 风格）

这里我只写你真正采用、也真正想讲的那种形式。设 `vae_encode()` 是图像到 VAE latent 的编码器，`net()` 是 DiT 主干，`sample_t()` 是时间采样器，`randn_like()` 采样高斯噪声。

JiT 风格 `x-pred` 的关键点是：

- 网络**直接输出** `x_pred`
- 再构造 `v_pred = (x_pred - z) / (1 - t)`
- 最后统一在 `v-loss` 上训练

伪代码如下：

```python
x = vae_encode(image)

t = sample_t()
eps = randn_like(x)
z = alpha(t) * x + sigma(t) * eps
v = (x - z) / (1 - t)

# network outputs clean latent directly
x_pred = net(z, t, cond)

# JiT-style reparameterization into v-space
v_pred = (x_pred - z) / (1 - t)

# optimize in v-space
loss = mse(v_pred, v)
```

这正是 JiT 代码里的核心结构：网络虽然直接输出 `x_pred`，但真正比较的是由它诱导出来的 `v_pred`。  
所以本文真正讨论的不是“`x-loss` 比 `eps-loss` 快”，而是：

> 在相同的 `v-loss` 框架下，让网络直接输出 `x`，会不会比直接输出 `eps` 或 `v` 更容易优化？

对 `eps-pred` / `v-pred`，本文只保留概念性对比：

- `eps-pred`：网络直接输出噪声对象；
- `v-pred`：网络直接输出混合对象；
- `x-pred`：网络直接输出 clean latent 对象。

本文的重点是第三种，也就是你真正观察到加速的那一种。

---

## 3. VAE latent 中的数据流形假设

下面进入本文最重要的结构假设。

我假设 VAE clean latent 不会占满整个 \(\mathbb{R}^D\)，而是集中在一个低维流形附近。为了方便写公式，先采用局部线性近似，把这个流形近似成一个 \(d\)-维子空间，其中 \(d \ll D\)。

于是可写成：

$$
x = U h,
\qquad
U \in \mathbb{R}^{D \times d},
\qquad
U^\top U = I_d.
$$

再设

$$
h \sim \mathcal{N}(0, \Lambda),
\qquad
\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_d),
$$

则 VAE clean latent 的协方差为

$$
\Sigma_x = U \Lambda U^\top.
$$

这表示：

> VAE clean latent 的主要变化集中在一个秩至多为 \(d\) 的低维子空间中。

而噪声不是这样。噪声仍然是：

$$
\epsilon \sim \mathcal{N}(0, I_D),
$$

也就是一个各向同性、近似满维的对象。

这一步决定了整个故事的根本不对称性：

- VAE clean latent 是低秩、结构化的；
- noise 是满秩、各向同性的。

VAE 已经先帮我们做了第一步工作：

> 它先把原始图像压进了一个更语义化、更低维、更结构化的表示空间。

而一旦在这个空间里再选 JiT 风格 `x-pred`，就等于要求 DiT 先回归这个 clean latent 本身，再把它投影到同一个 `v-loss` 上比较。

---

## 4. `x-pred` 的 Bayes 最优形式为什么天然低秩？

由

$$
z_t = \alpha_t x + \sigma_t \epsilon,
$$

可得 noisy latent 的协方差为

$$
\Sigma_z(t) = \alpha_t^2 \Sigma_x + \sigma_t^2 I_D.
$$

在高斯近似下，最优均方误差 `x-pred` 是条件期望：

$$
f_x^*(z_t)=\mathbb{E}[x\mid z_t]
= \alpha_t \Sigma_x \Sigma_z(t)^{-1} z_t.
$$

代入

$$
\Sigma_x = U\Lambda U^\top,
$$

得到

$$
f_x^*(z_t)
=
\alpha_t
U
\mathrm{diag}\!\left(
\frac{\lambda_i}{\alpha_t^2 \lambda_i + \sigma_t^2}
\right)
U^\top z_t.
$$

这个式子有一个非常关键的含义：

> `x-pred` 的最优映射只依赖于 \(z_t\) 在数据流形方向上的投影。

在流形正交补方向上，Bayes 最优 `x-pred` 自动为零。  
它不需要去恢复任何流形外信息。

因此：

$$
\mathrm{rank}(f_x^*) \le d.
$$

这说明 `x-pred` 本质上是一个低秩恢复任务。

更关键的是，在 JiT 风格写法里，虽然最终 loss 写在 `v-space`：

$$
v_\theta = \frac{x_\theta - z_t}{1-t},
$$

但网络本身仍然是先去回归这个低秩的 clean latent 对象。这才是优化更容易的根源。

---

## 5. `eps-pred` 和 `v-pred` 为什么仍然更接近满秩目标？

### 5.1 `eps-pred`

在高斯近似下，最优 `eps-pred` 为

$$
f_\epsilon^*(z_t)
=\mathbb{E}[\epsilon\mid z_t]
=\sigma_t \Sigma_z(t)^{-1} z_t.
$$

现在把空间分成两部分：

- 数据流形方向；
- 流形正交补方向。

在正交补空间中，clean signal 不存在，因此 noisy latent 完全由噪声主导。于是最优 `eps-pred` 在这些方向上仍然必须恢复噪声本身。

这意味着：

> `eps-pred` 不能像 `x-pred` 那样简单忽略流形外方向。  
> 它必须在几乎整个 ambient 空间里继续传递信息。

因此 `eps-pred` 的最优映射更接近满秩：

$$
\mathrm{rank}(f_\epsilon^*) \approx D.
$$

### 5.2 `v-pred`

现在看

$$
v_t = \beta_t x + \gamma_t \epsilon.
$$

其 Bayes 最优形式为

$$
f_v^*(z_t)
= \mathbb{E}[v_t\mid z_t]
= \beta_t \mathbb{E}[x\mid z_t] + \gamma_t \mathbb{E}[\epsilon\mid z_t].
$$

也就是

$$
f_v^*(z_t)=\beta_t f_x^*(z_t)+\gamma_t f_\epsilon^*(z_t).
$$

关键点在于：在流形正交补空间中，\(f_x^*\) 为零，而 \(f_\epsilon^*\) 不为零，因此

$$
f_v^*(z_t)|_{\perp} = \gamma_t P_\perp \epsilon.
$$

只要 \(\gamma_t \neq 0\)，也就是 `v` 中仍然含有噪声项，那么 `v-pred` 在结构上就仍然保留了流形外的噪声分量。

所以：

> `v-pred` 虽然比纯 `eps-pred` 更接近 clean signal，但它本质上仍然不是纯低秩恢复，而是一个带有满秩噪声分量的混合目标。

因此大致可以得到一个顺序：

$$
\text{`x-pred' easiest} \;<\; \text{`v-pred'} \;<\; \text{`eps-pred' hardest},
$$

其中 `v-pred` 的具体位置取决于噪声系数 \(\gamma_t\) 的大小。

---

## 6. 为什么这会直接变成训练步数差异？

上面的结论说的是“网络直接输出空间的几何不同”。  
而我真正观察到的是“训练步数不同”。

这里可以引入一个非常自然的优化观点：Transformer 有明显的谱偏置。也就是说，它更容易更早学会：

- 高能量主方向；
- 结构化、语义化的模式；
- 与网络特征更对齐的低秩成分。

如果把训练初期近似成一个线性化系统，那么不同模态上的误差衰减大致满足

$$
e_i^{(n)} \approx (1-\eta\mu_i)^n e_i^{(0)},
$$

其中 \(\mu_i\) 表示第 \(i\) 个模态的可学习速度。

这时，目标在哪些模态上有能量就很重要：

- 对 `x-pred` 来说，能量主要集中在数据流形主方向上；
- 对 `eps-pred` 来说，能量分散在大量 ambient 方向上；
- 对 `v-pred` 来说，虽然比 `eps-pred` 更好，但只要仍含噪声分量，就仍然会在更多弱模态上保留目标能量。

于是训练中就会发生：

> `x-pred` 把监督信号集中到模型最容易优先学会的少数主模态上；  
> `eps-pred` / `v-pred` 则要求模型在更多慢模态和流形外方向上继续分配拟合预算。

这就解释了为什么：

> 在同样的 `v-loss` 框架下，仅仅把网络直接输出从 `eps/v` 换成 `x`，训练步数就会显著减少。

---

## 7. 为什么 VAE latent 会放大这个优势？

如果 `x-pred` 本身在几何上已经更合理，那么为什么一旦把它放到 **VAE latent** 上，收益会进一步变大？

因为 VAE 编码器对 clean signal 和 noise 的作用并不对称。

一个好的 VAE encoder 往往会：

- 压掉像素级局部冗余；
- 把语义变化集中到更少主方向；
- 让 clean latent 的协方差谱更陡；
- 让 clean latent 更接近低维语义流形。

于是：

- clean signal 的有效维度下降；
- `x-pred` 的低秩性更明显；
- 但噪声和噪声混合目标并不会等比例变成低秩。

换句话说：

> VAE 会压缩数据流形，但不会消灭 `eps/v` 目标中的噪声分量。

因此在 VAE latent 中，`x-pred` 和 `eps/v-pred` 的结构差异会更明显。

这就是为什么我观察到的，不是一个小幅收益，而是可能达到 2 倍、3 倍这种级别的训练步数缩短。

---

## 8. 为什么 3 倍是完全合理的？

这套解释不会严格推出“恰好 3 倍”，也不需要这样做。  
它真正说明的是：只要满足以下条件，显著大于 1 的步数缩短就是完全合理的。

- VAE clean latent 的有效秩远小于 ambient 维度；
- DiT 处在中等容量区间，足以快速拟合数据流形，但不足以轻松覆盖满维带噪目标；
- 生成质量主要由先学会的主模态决定；
- `eps-pred` / `v-pred` 在大量慢模态上仍需继续付出拟合成本。

这时会自然出现这样的现象：

- `x-pred` 很早就学会了有用结构；
- `v-pred` 次之，因为它仍要处理一部分噪声成分；
- `eps-pred` 最慢，因为它几乎就是在做满维噪声重建。

所以“用大约三分之一训练步数达到同等结果”并不神秘。  
它只是说明：

> 网络直接输出空间被改对了。

---

## 9. 一个最值得记住的结论

如果把整件事压成一句话，我会这样说：

> VAE latent + JiT 风格 `x-pred` 的本质，不是换了一个更顺手的 loss，而是把 DiT 的直接回归对象从“带噪目标”改成了“clean latent”。

或者写得更短一点：

> 在同样的 `v-loss` 框架下，`x-pred` 参数化让 DiT 不再优先学习噪声，而是优先学习 VAE latent 中的数据流形。

我认为这就是它为什么会显著减少训练步数的最本质解释。

---

## 10. 可继续验证的预测

如果这套解释是对的，那么它还会导出几个很自然的实验预测：

- VAE latent 越语义化、越低秩，`x-pred` 相对于 `eps/v-pred` 的步数优势越大；
- 模型越瓶颈化，`x-pred` 相对于 `eps/v-pred` 的优势越明显；
- 这个差距主要体现于训练前中期的 sample efficiency，而不一定只体现于无限训练时间下的最终最优值；
- `x-pred` 的梯度方差应当低于 `eps-pred` / `v-pred`；
- 如果显式把目标投影到主 latent 子空间，那么 `x-pred` 与 `eps/v-pred` 的差距应当缩小。

这些实验如果都成立，那么“VAE latent + JiT 风格 `x-pred` = 低秩流形恢复”这个解释就会变得非常有说服力。
