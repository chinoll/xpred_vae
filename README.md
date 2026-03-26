**GitHub渲染有错误，自行下载PDF查看**

# 当 x-prediction 回归 Latent Space：被忽视的维度差红利

> **摘要：** JiT (Li & He, 2025) 基于流形假设证明了像素空间扩散模型中 x-prediction 的必要性——$\epsilon$/$v$-prediction 在高维 patch 下灾难性崩溃。这一发现让大 patch + plain ViT 的像素空间训练成为可能。然而，学界随之形成了一个隐含假设：x-prediction 是像素空间的专属方案，latent space 维度已经足够低，$\epsilon$/$v$-prediction "够用了"。本文对这一假设提出质疑。我们将 x-prediction 引入 latent space 扩散模型，观察到训练拟合效率获得**数倍提升**——不改架构、不加参数、不增单步计算量。下文从流形几何出发，沿用 JiT 论文的数学框架，给出严格的理论分析。

---

## 1. 引言：像素空间扩散的两条路线与 Latent Space 的盲区

在 latent diffusion model 统治图像生成的今天，像素空间扩散一直是一条值得关注的替代路线——它不依赖预训练 VAE tokenizer，是真正自包含的生成范式。近年来，像素空间扩散走出了两条不同的技术路线：

**路线一：Flow Matching + 架构补丁。** 以 Simple Diffusion (Hoogeboom et al., 2023)、PixelFlow (Chen et al., 2025) 等为代表。沿用 $\epsilon$/$v$-prediction（即 flow matching 的标准做法），但引入密集卷积、小 patch、长跳跃连接、多尺度架构、pixel decoder 等设计来"硬扛"高维噪声预测的信息瓶颈。

**路线二：x-prediction + Plain ViT。** 以 JiT (Li & He, 2025) 为代表。从流形假设出发，指出问题根源不在架构而在预测目标——让网络直接预测干净图像 $x$ 而非 $\epsilon$ 或 $v$，就能用朴素 ViT 在 patch size 16 甚至 32 的像素空间上训练。不需要 tokenizer、额外 loss 或 pixel decoder。

JiT 的实验结论令人印象深刻：在 ImageNet 256×256、JiT-B/16 配置下，x-prediction 取得 FID 8.62，而 $\epsilon$-prediction 的 FID 高达 394，$v$-prediction 达 96——灾难性崩溃。这证明在高维像素空间中，**预测目标的选择比架构设计重要得多**。

但 JiT 论文明确提到了一个关键观察（其原文 Table 2(b)）：在 JiT-B/4、64×64 分辨率下（每个 patch 仅 48 维，远小于 hidden size 768），九种预测/损失组合的差异变得微不足道。论文据此指出：

> *"Many previous latent diffusion models have a similarly small input dimensionality and therefore were not exposed to the issue we discuss here."*

这句话被广泛理解为："latent space 维度够低，x-prediction 的优势不存在了。" 但这个推论**并不成立**。维度够低意味着不会崩溃，但**不意味着最优**。

一个自然的问题被忽略了：**即使 $\epsilon$/$v$-prediction 在 latent space 中"不崩溃"，x-prediction 是否仍然更高效？**

---

## 2. 形式化框架：从 JiT 到 Latent Space

### 2.1 回顾 JiT 的核心框架

JiT 论文建立了一个清晰的数学框架来分析扩散模型中的预测目标选择。我们完整地沿用这一框架并将其推广到 latent space。

考虑线性插值噪声调度 (Lipman et al., 2022; Liu et al., 2022)。给定干净数据 $\boldsymbol{x} \sim p_{\text{data}}$ 和噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$，噪声样本为：

$$\boldsymbol{z}_t = t \, \boldsymbol{x} + (1 - t) \, \boldsymbol{\epsilon}, \quad t \in [0, 1] \tag{1}$$

其中 $t = 1$ 对应干净数据，$t = 0$ 对应纯噪声。流速度定义为 $\boldsymbol{z}_t$ 对时间的导数：

$$\boldsymbol{v} = \boldsymbol{x} - \boldsymbol{\epsilon} \tag{2}$$

给定三个未知量 $\{\boldsymbol{x}, \boldsymbol{\epsilon}, \boldsymbol{v}\}$ 和两个约束（式(1)和式(2)），只要网络输出其中任一个，其余两个均可由代数关系推出。JiT 论文系统地列举了**九种可能的组合**——三种预测空间（网络直接输出 $\boldsymbol{x}$、$\boldsymbol{\epsilon}$ 或 $\boldsymbol{v}$）× 三种损失空间（损失定义在 $\boldsymbol{x}$、$\boldsymbol{\epsilon}$ 或 $\boldsymbol{v}$ 上）。

以 **x-prediction** 为例。设网络直接输出 $\boldsymbol{x}_\theta = \text{net}_\theta(\boldsymbol{z}_t, t)$，则由约束关系可推出：

$$\boldsymbol{\epsilon}_\theta = \frac{\boldsymbol{z}_t - t \, \boldsymbol{x}_\theta}{1 - t}, \qquad \boldsymbol{v}_\theta = \frac{\boldsymbol{x}_\theta - \boldsymbol{z}_t}{1 - t} \tag{3}$$

类似地，若网络直接输出 $\boldsymbol{\epsilon}$ 或 $\boldsymbol{v}$，则 $\boldsymbol{x}$ 需要通过如下变换还原：

- $\boldsymbol{\epsilon}$-prediction → $\boldsymbol{x}$：$\boldsymbol{x}_\theta = \frac{\boldsymbol{z}_t - (1-t)\,\boldsymbol{\epsilon}_\theta}{t}$
- $\boldsymbol{v}$-prediction → $\boldsymbol{x}$：$\boldsymbol{x}_\theta = (1-t)\,\boldsymbol{v}_\theta + \boldsymbol{z}_t$

### 2.2 JiT 的算法选择及其数学含义

JiT 最终选择了 **x-prediction + v-loss**。展开写为：

$$\mathcal{L} = \mathbb{E}_{t, \boldsymbol{x}, \boldsymbol{\epsilon}} \left\| \boldsymbol{v}_\theta(\boldsymbol{z}_t, t) - \boldsymbol{v} \right\|^2, \quad \text{where} \quad \boldsymbol{v}_\theta = \frac{\text{net}_\theta(\boldsymbol{z}_t, t) - \boldsymbol{z}_t}{1 - t} \tag{4}$$

将 $\boldsymbol{v} = (\boldsymbol{x} - \boldsymbol{z}_t)/(1-t)$ 和 $\boldsymbol{v}_\theta = (\boldsymbol{x}_\theta - \boldsymbol{z}_t)/(1-t)$ 代入，得到：

$$\mathcal{L} = \mathbb{E}_{t, \boldsymbol{x}, \boldsymbol{\epsilon}} \frac{1}{(1-t)^2} \left\| \boldsymbol{x}_\theta - \boldsymbol{x} \right\|^2 \tag{5}$$

这揭示了 v-loss + x-prediction 的本质：它等价于一个**时间步加权的 x-loss**，权重为 $w(t) = 1/(1-t)^2$。当 $t \to 0$（高噪声）时权重趋近 1；当 $t \to 1$（低噪声）时权重趋近 $+\infty$。这意味着损失函数在接近干净数据的区间施加更大的惩罚——直觉上，当图像已经"基本成型"时，精细结构的微小偏差代价更高。

### 2.3 将框架迁移到 Latent Space

现在设 $\boldsymbol{x}_l = \text{Enc}(\boldsymbol{x}) \in \mathbb{R}^{D_l}$ 为 VAE encoder 的输出。**以上所有公式完全不变**——只需将 $\boldsymbol{x}$ 替换为 $\boldsymbol{x}_l$，将工作空间从 $\mathbb{R}^D$ 替换为 $\mathbb{R}^{D_l}$：

$$\boldsymbol{z}_t = t \, \boldsymbol{x}_l + (1 - t) \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_{D_l}) \tag{6}$$

$$\mathcal{L}_{\text{latent}} = \mathbb{E}_{t, \boldsymbol{x}_l, \boldsymbol{\epsilon}} \frac{1}{(1-t)^2} \left\| \text{net}_\theta(\boldsymbol{z}_t, t) - \boldsymbol{x}_l \right\|^2 \tag{7}$$

九宫格的结构、推导、ODE 采样过程均完全平行。唯一的区别在于**维度**：$D_l \ll D$。

JiT 论文在 Table 2(b) 观察到的"48 维时九种组合差异不大"暗示了一个重要推论：$\epsilon$/$v$-prediction 的性能退化是**维度的连续函数**，而非一个非此即彼的门槛效应。当 $D_l$ 从 48 增大到 4096（典型 latent space），退化并非消失——只是从"灾难性"变为"隐性"。

---

## 3. 数学分析：为什么 x-prediction 在 Latent Space 更高效

### 3.1 流形假设与三个维度层级

JiT 的整个论证建立在流形假设 (Chapelle et al., 2006) 上。我们将其精确化。

**定义 (流形假设).** 设自然图像数据分布 $p_{\text{data}}$ 的支撑集 $\text{supp}(p_{\text{data}})$ 近似为一个 $d$ 维紧致黎曼流形 $\mathcal{M}$，嵌入在像素空间 $\mathbb{R}^D$ 中，其中 $d \ll D$。

经过 VAE 编码后，latent code 分布在像流形 $\mathcal{M}_l = \text{Enc}(\mathcal{M})$ 上，$\mathcal{M}_l$ 是 $\mathbb{R}^{D_l}$ 中的 $d$ 维子流形（编码器是光滑映射，保持流形维度不变）。

因此存在三个维度层级：

$$d \ll D_l \ll D$$

具体地，对于 ImageNet 256×256 + SD-VAE 的典型配置：

| 量 | 含义 | 值 |
|---|------|---|
| $D$ | 像素空间维度 | 196,608 ($= 256 \times 256 \times 3$) |
| $D_l$ | Latent 空间维度 | 4,096 ($= 32 \times 32 \times 4$) |
| $d$ | 流形内禀维度 | $\sim 10^2$ — $10^3$ (经验估计) |

### 3.2 最优去噪函数的结构

给定噪声样本 $\boldsymbol{z}_t$，任何预测目标的最优解（Bayes 最优）都可以用后验期望表示。

**x-prediction 的 Bayes 最优解** 为后验均值：

$$\boldsymbol{x}_l^*(z_t, t) = \mathbb{E}[\boldsymbol{x}_l \mid \boldsymbol{z}_t, t] \tag{8}$$

这与经典的 Tweedie 公式直接相关。由于 $\boldsymbol{x}_l$ 分布在 $d$ 维流形 $\mathcal{M}_l$ 上，后验均值 $\boldsymbol{x}_l^*$ 也（近似地）在 $\mathcal{M}_l$ 上或其邻域。换言之，**最优 x-prediction 函数的像集是 $d$ 维的**。

**$\epsilon$-prediction 的 Bayes 最优解** 为：

$$\boldsymbol{\epsilon}^*(\boldsymbol{z}_t, t) = \mathbb{E}[\boldsymbol{\epsilon} \mid \boldsymbol{z}_t, t] = \frac{\boldsymbol{z}_t - t \, \boldsymbol{x}_l^*(\boldsymbol{z}_t, t)}{1 - t} \tag{9}$$

虽然 $\boldsymbol{\epsilon}^*$ 可以从 $\boldsymbol{x}_l^*$ 推出，但 $\boldsymbol{\epsilon}^*$ 本身是一个 $D_l$ 维的向量场——它在 $\mathbb{R}^{D_l}$ 的所有方向上都有非零分量。即使数据在低维流形上，**最优噪声预测函数必须在所有 $D_l$ 个方向上提供精确值**。

**v-prediction（flow matching 的标准形式）的 Bayes 最优解** 为：

$$\boldsymbol{v}^*(\boldsymbol{z}_t, t) = \mathbb{E}[\boldsymbol{v} \mid \boldsymbol{z}_t, t] = \mathbb{E}[\boldsymbol{x}_l - \boldsymbol{\epsilon} \mid \boldsymbol{z}_t, t] = \boldsymbol{x}_l^*(\boldsymbol{z}_t, t) - \boldsymbol{\epsilon}^*(\boldsymbol{z}_t, t) \tag{9b}$$

这是 flow matching (Lipman et al., 2022; Liu et al., 2022) 以及 SD3 (Esser et al., 2024) 等方法的训练目标。$\boldsymbol{v}^*$ 是**两个不同维度结构的量的差**：$\boldsymbol{x}_l^*$ 在 $d$ 维流形上，$\boldsymbol{\epsilon}^*$ 在 $D_l$ 维空间中。它们的差 $\boldsymbol{v}^*$ 不在任何低维子空间中——它继承了 $\boldsymbol{\epsilon}^*$ 的全维性。

更直观地理解：在任一固定时间步 $t$ 和固定 $\boldsymbol{z}_t$ 下，后验分布 $p(\boldsymbol{x}_l | \boldsymbol{z}_t)$ 和 $p(\boldsymbol{\epsilon} | \boldsymbol{z}_t)$ 之间存在确定性约束 $\boldsymbol{z}_t = t\boldsymbol{x}_l + (1-t)\boldsymbol{\epsilon}$。但 $\boldsymbol{v} = \boldsymbol{x}_l - \boldsymbol{\epsilon}$ 的后验均值仍然需要在 $D_l$ 维空间中精确指定——因为 $\boldsymbol{\epsilon}$ 的各向同性性质"污染"了所有方向。

这就是核心不对称性：
- **x-prediction 学习的是从 $\mathbb{R}^{D_l}$ 到 $d$ 维流形的投影**——目标空间是低维的
- **$\epsilon$-prediction 学习的是从 $\mathbb{R}^{D_l}$ 到 $\mathbb{R}^{D_l}$ 的映射**——目标空间是全维的
- **v-prediction (flow matching) 学习的也是 $\mathbb{R}^{D_l}$ 到 $\mathbb{R}^{D_l}$ 的映射**——尽管 $\boldsymbol{v}$ 表面上连接了 $\boldsymbol{x}$ 和 $\boldsymbol{\epsilon}$，但它本身仍然是全维的

### 3.3 逼近误差的维度依赖性

我们援引流形上函数逼近的经典结论 (Nakada & Imaizumi, 2020; Chen et al., 2022; Schmidt-Hieber, 2019)。

**定理 (流形自适应逼近, 非形式化).** 设 $f: \mathcal{M} \to \mathbb{R}^m$ 为定义在 $d$ 维紧致黎曼流形 $\mathcal{M} \hookrightarrow \mathbb{R}^n$ 上的 $(s, C)$-Hölder 光滑函数（$s > 0$ 为光滑度，$C$ 为 Hölder 常数）。则存在具有 $N$ 个参数的 ReLU 网络 $\hat{f}_N$，使得：

$$\left\| \hat{f}_N - f \right\|_{L^2(\mathcal{M})}^2 = O\left( C^2 \cdot N^{-2s/d} \right) \tag{10}$$

**其中 $d$ 为流形 $\mathcal{M}$ 的内禀维度，与环境空间维度 $n$ 无关。**

这个定理是理解 x-prediction 优势的数学基石。它告诉我们：如果学习目标的"有效维度"是 $d$，那么无论它嵌入在多高维的空间中，所需的网络容量只与 $d$ 有关。

现在将其应用到去噪任务。设网络参数量为 $N$。

**x-prediction 的逼近误差.** 目标函数 $\boldsymbol{x}_l^*(\boldsymbol{z}_t, t)$ 的像集在 $d$ 维流形 $\mathcal{M}_l$ 上。由式(10)：

$$\mathcal{E}_x(N) = \mathbb{E}\left\| \hat{\boldsymbol{x}}_\theta - \boldsymbol{x}_l^* \right\|^2 = O\left(N^{-2s/d}\right) \tag{11}$$

**$\epsilon$-prediction 的逼近误差.** 目标函数 $\boldsymbol{\epsilon}^*(\boldsymbol{z}_t, t)$ 是 $\mathbb{R}^{D_l}$ 上的全维函数。网络需要在 $D_l$ 个独立方向上同时逼近：

$$\mathcal{E}_\epsilon(N) = \mathbb{E}\left\| \hat{\boldsymbol{\epsilon}}_\theta - \boldsymbol{\epsilon}^* \right\|^2 = O\left(N^{-2s/D_l}\right) \tag{12}$$

**v-prediction (flow matching) 的逼近误差.** 这是当前 latent diffusion 的主流训练范式（SD3、FLUX、SiT 均采用此形式）。目标函数 $\boldsymbol{v}^*(\boldsymbol{z}_t, t) = \boldsymbol{x}_l^* - \boldsymbol{\epsilon}^*$ 是两个不同维度结构的函数之差。我们可以精确分析其逼近复杂度。

将 $\boldsymbol{v}^*$ 在 latent 协方差矩阵 $\boldsymbol{\Sigma}_l$ 的特征方向上分解。对于第 $j$ 个特征方向（对应特征值 $\lambda_j$），$\boldsymbol{v}^*$ 在该方向上的分量为：

$$v_j^* = x_{l,j}^* - \epsilon_j^* \tag{12b}$$

$x_{l,j}^*$ 的变化幅度正比于 $\sqrt{\lambda_j}$（数据方差），而 $\epsilon_j^*$ 在所有方向上的变化幅度相同（噪声各向同性）。在 $\lambda_j \approx 0$ 的方向上（即流形的法方向），$v_j^* \approx -\epsilon_j^*$，仍然需要精确预测。因此 $\boldsymbol{v}^*$ 的有效维度等于 $D_l$，与 $\epsilon$-prediction 相同：

$$\mathcal{E}_v(N) = \mathbb{E}\left\| \hat{\boldsymbol{v}}_\theta - \boldsymbol{v}^* \right\|^2 = O\left(N^{-2s/D_l}\right) \tag{12c}$$

**这是一个重要的澄清：** 尽管 flow matching 的 $\boldsymbol{v} = \boldsymbol{x} - \boldsymbol{\epsilon}$ 在形式上"包含"了干净数据 $\boldsymbol{x}$，但从逼近论的角度看，$v$-prediction 的复杂度与 $\epsilon$-prediction 完全相同——都受 $D_l$ 维环境空间的约束。$\boldsymbol{v}$ 中的 $\boldsymbol{x}$ 分量虽然是低维的，但 $-\boldsymbol{\epsilon}$ 分量的全维性使得整体逼近速率无法改善。

**效率比.** 要达到相同目标误差 $\delta$，三种方案所需参数量分别为：

$$N_x \sim \delta^{-d/(2s)}, \qquad N_\epsilon \sim N_v \sim \delta^{-D_l/(2s)} \tag{13}$$

x-prediction 与 $\epsilon$/$v$-prediction 之比为：

$$\boxed{\frac{N_\epsilon}{N_x} = \frac{N_v}{N_x} \sim \delta^{-(D_l - d)/(2s)}} \tag{14}$$

这是**超多项式**的增长。$\epsilon$-prediction 和 v-prediction (flow matching) 在逼近效率上完全等价——它们面临的是同一个维度诅咒。对于 $D_l = 4096$，$d = 500$，即使 $s = 2$（二阶光滑），指数差 $(D_l - d)/(2s) = 899$ 意味着 x-prediction 的参数效率在理论上存在巨大优势。

当然，实际训练中的增益不会达到理论极限——式(10)中的常数项、网络架构的具体选择、优化器的行为都会影响实际增益。但**方向是明确的**：x-prediction 在 latent space 中对比 flow matching ($v$-pred) 和 $\epsilon$-pred 都拥有系统性的逼近优势。

### 3.4 误差放大分析：从预测噪声到还原图像

即使我们不关心逼近理论的渐近结论，一个更直接的分析也能揭示 $\epsilon$/$v$-prediction 的固有劣势：**误差放大效应**。

在生成过程中，无论使用哪种预测目标，最终都需要将网络输出转换到 $\boldsymbol{v}$ 空间来求解 ODE（式(4)）。考虑 $\epsilon$-prediction 的情况。由 JiT 的九宫格（列(b)行(3)），从 $\boldsymbol{\epsilon}_\theta$ 到 $\boldsymbol{v}_\theta$ 的转换为：

$$\boldsymbol{v}_\theta = \frac{\boldsymbol{z}_t - \boldsymbol{\epsilon}_\theta}{t} \tag{15}$$

设 $\epsilon$-prediction 的残差为 $\Delta\boldsymbol{\epsilon} = \boldsymbol{\epsilon}_\theta - \boldsymbol{\epsilon}$，则由式(1)和式(2)推出 $\boldsymbol{x}$ 的重建误差为：

$$\Delta\boldsymbol{x} = \boldsymbol{x}_\theta - \boldsymbol{x} = -\frac{1 - t}{t} \Delta\boldsymbol{\epsilon} \tag{16}$$

**误差放大系数** $|(1-t)/t|$ 在不同时间步下的行为如下：

| 时间步 $t$ | $(1-t)/t$ | 含义 |
|-----------|-----------|------|
| $t = 0.9$（低噪声） | 0.11 | 误差衰减 |
| $t = 0.5$（中等噪声） | 1.0 | 不放大 |
| $t = 0.1$（高噪声） | 9.0 | **9 倍放大** |
| $t = 0.01$ | 99.0 | **99 倍放大** |

在高噪声区间（$t$ 接近 0），$\epsilon$-prediction 中的任何预测误差都会被放大 $(1-t)/t$ 倍后传递到 $\boldsymbol{x}$ 的重建中。而正是 JiT 所使用的 logit-normal 分布（$\mu = -0.8$）将采样集中在高噪声区间。

类似地，对于 $v$-prediction，从 $\boldsymbol{v}_\theta$ 到 $\boldsymbol{x}$ 的转换为 $\boldsymbol{x}_\theta = (1-t)\boldsymbol{v}_\theta + \boldsymbol{z}_t$，误差为：

$$\Delta\boldsymbol{x} = (1-t) \Delta\boldsymbol{v} \tag{17}$$

虽然这里的系数 $(1-t) \leq 1$ 看似没有放大效应，但问题在于 $\boldsymbol{v} = \boldsymbol{x} - \boldsymbol{\epsilon}$ 本身是一个 $D_l$ 维的量（$\boldsymbol{x}$ 在流形上但 $\boldsymbol{\epsilon}$ 充满全空间），网络首先需要在 $D_l$ 维空间中精确预测 $\boldsymbol{v}$，然后才能通过系数衰减。

**对于 x-prediction**，不存在任何转换环节——$\boldsymbol{x}_\theta = \text{net}_\theta(\boldsymbol{z}_t, t)$ 直接就是最终输出，$\Delta\boldsymbol{x} = \text{net}_\theta - \boldsymbol{x}_l$。误差就是网络逼近误差本身，不经过任何放大或变换。

### 3.5 残差向量的几何结构与梯度信噪比

以上分析关注的是"最终能达到的精度"。但训练效率更关心的问题是：**每一步梯度下降学到了多少？**

考虑 v-loss 下的单样本梯度。对于 x-prediction 和 $\epsilon$-prediction 两种参数化，损失函数分别为：

$$\mathcal{L}_x = \frac{1}{(1-t)^2} \|\boldsymbol{x}_\theta - \boldsymbol{x}_l\|^2, \qquad \mathcal{L}_\epsilon = \|\boldsymbol{\epsilon}_\theta - \boldsymbol{\epsilon}\|^2 \tag{18}$$

（两者都是 v-loss 在不同参数化下的等价形式）

关键在于**残差向量的几何结构**：

**x-prediction 的残差.** $\boldsymbol{r}_x = \boldsymbol{x}_\theta - \boldsymbol{x}_l$。由于 $\boldsymbol{x}_l \in \mathcal{M}_l$，且经过足够训练后 $\boldsymbol{x}_\theta$ 也近似在 $\mathcal{M}_l$ 附近，残差 $\boldsymbol{r}_x$ 主要集中在 $\boldsymbol{x}_l$ 处流形 $\mathcal{M}_l$ 的**切空间** $T_{\boldsymbol{x}_l}\mathcal{M}_l$ 中，这是一个 $d$ 维子空间。

更严格地，设 $\Pi_{\boldsymbol{x}_l}: \mathbb{R}^{D_l} \to T_{\boldsymbol{x}_l}\mathcal{M}_l$ 为正交投影算子，则：

$$\boldsymbol{r}_x \approx \Pi_{\boldsymbol{x}_l} \boldsymbol{r}_x + O(\|\boldsymbol{r}_x\|^2 / \tau) \tag{19}$$

其中 $\tau$ 为流形的 reach（流形到其中轴的最小距离）。残差的能量集中在 $d$ 个方向上。

**$\epsilon$-prediction 的残差.** $\boldsymbol{r}_\epsilon = \boldsymbol{\epsilon}_\theta - \boldsymbol{\epsilon}$。由于真实噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_{D_l})$ 在 $D_l$ 维空间中各向同性分布，残差 $\boldsymbol{r}_\epsilon$ 的能量在所有 $D_l$ 个方向上大致均匀分布。不存在任何低维子空间的集中现象。

**梯度信噪比.** 参数梯度为 $\boldsymbol{g}_\theta = \boldsymbol{r}^\top \nabla_\theta \text{net}_\theta$。对于 minibatch SGD，梯度的有效性取决于残差方向的一致性。

在 x-prediction 中，不同样本的残差都集中在各自的切空间中——虽然切空间因点而异，但它们共享流形 $\mathcal{M}_l$ 的全局结构。这意味着 minibatch 内的梯度有 $d$ 个"有效维度"，信号集中度为：

$$\text{SNR}_x \propto \frac{\|\boldsymbol{r}_x\|^2}{d} \tag{20}$$

在 $\epsilon$-prediction 中，不同样本的残差在 $D_l$ 维空间中近似各向同性，梯度信号被稀释到 $D_l$ 个方向：

$$\text{SNR}_\epsilon \propto \frac{\|\boldsymbol{r}_\epsilon\|^2}{D_l} \tag{21}$$

**在可比较的总残差量级下，每步梯度的有效信噪比之比为：**

$$\boxed{\frac{\text{SNR}_x}{\text{SNR}_\epsilon} \approx \frac{D_l}{d}} \tag{22}$$

对于 $D_l = 4096$、$d \sim 500$ 的典型 latent space，这个比值约为 $8\times$，与我们观察到的训练效率提升在量级上一致。

### 3.6 Fisher 信息矩阵与损失景观条件数

上述分析可以在更严格的优化理论框架中形式化。定义 Fisher 信息矩阵：

$$\boldsymbol{F}(\theta) = \mathbb{E}_{t, \boldsymbol{x}_l, \boldsymbol{\epsilon}} \left[ \nabla_\theta \ell \cdot \nabla_\theta \ell^\top \right] \tag{23}$$

其特征值谱决定了一阶优化器（如 Adam）的收敛性。

**$\epsilon$-prediction 的 Fisher 谱.** $\boldsymbol{F}$ 的特征值跨越 $D_l$ 个输出维度。其中 $d$ 个特征值对应流形方向上的噪声分量，与数据结构关联紧密，量级较大；$D_l - d$ 个特征值对应正交方向上的噪声分量，它们对最终生成质量的贡献较小，但仍然参与优化动力学。这导致 $\boldsymbol{F}$ 的有效条件数被拉大。

**x-prediction 的 Fisher 谱.** 预测目标在 $d$ 维流形上，$\boldsymbol{F}$ 只有 $d$ 个显著特征值（对应流形切方向）。其余 $D_l - d$ 个特征值趋近于零，不参与优化动力学。有效条件数仅由流形自身的几何（曲率）决定：

$$\kappa_{\text{eff}}^{(x)} \sim \kappa(\mathcal{M}_l), \qquad \kappa_{\text{eff}}^{(\epsilon)} \sim \kappa(\mathcal{M}_l) \cdot \frac{D_l}{d} \tag{24}$$

这里引出了一个额外的增益来源：**VAE 作为几何预条件器**。

VAE encoder 的训练目标（重建 + KL 正则化）隐含地鼓励 latent 空间具有良好的几何性质。KL 正则项将 latent 分布推向各向同性高斯，这等价于"展平"流形——像素空间中高度弯曲、自相交的图像流形 $\mathcal{M}$，在 latent space 中被映射为曲率更低、reach $\tau(\mathcal{M}_l)$ 更大的流形 $\mathcal{M}_l$。

**这种几何预条件化的好处，x-prediction 能享受到，$\epsilon$-prediction 不能。** 因为 x-prediction 的优化难度（条件数）只取决于 $\kappa(\mathcal{M}_l)$，而 $\epsilon$-prediction 的条件数被 $D_l/d$ 因子支配——无论流形被展平到什么程度，噪声预测的维度负担不变。

### 3.7 对 JiT toy experiment 的推广：连续退化而非二值跳变

JiT 论文的 toy experiment 设置如下：$d = 2$ 维的数据通过随机正交投影矩阵 $\boldsymbol{P} \in \mathbb{R}^{D \times d}$ 嵌入到 $D$ 维空间中。用 5 层 ReLU MLP（256 维 hidden units）训练生成模型。结论是：$D = 2$ 时三种预测都行；$D = 16$ 时 $\epsilon$/$v$-prediction 开始挣扎；$D = 512$ 时灾难性失败。

这个实验揭示的关键规律是：**预测难度是环境维度 $D$ 与隐藏维度 $h$ 之比的函数**。当 $D/h$ 从 $< 1$ 增大到 $\gg 1$ 时，$\epsilon$/$v$-prediction 的性能连续退化。

现在考虑 latent space 的情况。设 $D_l = 4096$，$d = 500$，DiT-XL 的 hidden dimension $h = 1152$。则：

$$\frac{D_l}{h} = \frac{4096}{1152} \approx 3.6 \tag{25}$$

这个比值处于 JiT toy experiment 中"开始退化但尚未崩溃"的区间（类比 $D/h = 16/256 = 0.06$ 到 $512/256 = 2$）。网络"能做"，但**远非最优**。

对于 x-prediction，有效比值为：

$$\frac{d}{h} = \frac{500}{1152} \approx 0.43 \tag{26}$$

这完全处于"所有方法都工作良好"的安全区间，对应 JiT 的 Table 2(b) 情景。

**核心论点：** latent space 中的 $\epsilon$/$v$-prediction 对应 JiT toy experiment 中 $D/h \approx 3\text{–}4$ 的区间——不会崩溃，但已经在为不必要的高维负担付出隐性代价。x-prediction 将有效比值降至 $d/h < 1$，回到最优区间。

---

## 4. VAE Latent Space 的几何结构与 x-prediction 的协同

前文的分析建立在一个关键假设上：VAE encoder 将像素空间中的数据流形 $\mathcal{M}$ 映射为 latent space 中的流形 $\mathcal{M}_l$，且 $\mathcal{M}_l$ 具有"更好"的几何性质。这一节我们严格地分析 VAE 的训练目标如何塑造 latent space 的几何结构，以及为什么这种结构与 x-prediction 产生协同效应而与 $\epsilon$/$v$-prediction 无关。

### 4.1 VAE 的优化目标与 Latent Space 的诱导结构

VAE (Kingma & Welling, 2013) 通过最大化证据下界 (ELBO) 来训练：

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log p_\psi(\boldsymbol{x} | \boldsymbol{z}) \right] - D_{\text{KL}}\left( q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}) \right) \tag{27}$$

其中 $q_\phi(\boldsymbol{z}|\boldsymbol{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\boldsymbol{x}), \text{diag}(\boldsymbol{\sigma}_\phi^2(\boldsymbol{x})))$ 是编码器的后验近似，$p(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_{D_l})$ 是先验。这两项分别施加了两种几何约束：

**重建项** $\mathbb{E}[\log p_\psi(\boldsymbol{x}|\boldsymbol{z})]$ 要求 latent code $\boldsymbol{z}$ 保留足够的信息以重建 $\boldsymbol{x}$。这等价于要求编码映射 $\text{Enc}: \mathcal{M} \to \mathcal{M}_l$ 近似保持流形的拓扑结构——不同的输入图像必须映射到不同的 latent code（否则无法重建）。从微分几何的角度，这要求 encoder 的 Jacobian $\boldsymbol{J}_\phi(\boldsymbol{x}) = \partial \boldsymbol{\mu}_\phi / \partial \boldsymbol{x} \in \mathbb{R}^{D_l \times D}$ 在流形的切方向上满秩。

**KL 项** $D_{\text{KL}}(q_\phi \| p)$ 将后验推向标准高斯先验。对于单个样本 $\boldsymbol{x}$，KL 散度展开为：

$$D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^{D_l} \left( \mu_{\phi,j}^2 + \sigma_{\phi,j}^2 - \log \sigma_{\phi,j}^2 - 1 \right) \tag{28}$$

这一项对 latent space 的几何产生三个关键效应：

**(a) 均值正则化：** $\mu_{\phi,j}^2$ 项惩罚 latent code 偏离原点，防止流形 $\mathcal{M}_l$ 在 $\mathbb{R}^{D_l}$ 中漂移到远离原点的区域。这意味着 $\mathcal{M}_l$ 被约束在以原点为中心的有界区域内。

**(b) 方差正则化：** $\sigma_{\phi,j}^2 - \log \sigma_{\phi,j}^2 - 1$ 的最小值在 $\sigma_{\phi,j}^2 = 1$ 处取得。这鼓励每个 latent 维度的局部方差接近 1——即流形 $\mathcal{M}_l$ 在每个坐标方向上具有大致均匀的"厚度"。

**(c) 维度利用压力：** KL 项对每个维度 $j$ 独立施加惩罚。如果某个维度 $j$ 对重建没有贡献（即 $\mu_{\phi,j} \approx 0, \sigma_{\phi,j} \approx 1$ 对所有 $\boldsymbol{x}$），则该维度的 KL 代价为零。反之，任何"被使用"的维度（$\mu_{\phi,j}$ 因 $\boldsymbol{x}$ 而变化）都会产生非零 KL 代价。这构成了一个**隐式的维度选择机制**：VAE 只会"激活"重建所必需的维度，其余维度保持为先验噪声。

### 4.2 Latent 协方差矩阵的谱结构

上述分析可以通过 latent code 的经验协方差矩阵精确量化。设 $\boldsymbol{x}_l = \boldsymbol{\mu}_\phi(\boldsymbol{x})$ 为 encoder 均值（实际使用时的 latent code），定义经验协方差：

$$\boldsymbol{\Sigma}_l = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}} \left[ (\boldsymbol{x}_l - \bar{\boldsymbol{x}}_l)(\boldsymbol{x}_l - \bar{\boldsymbol{x}}_l)^\top \right] \in \mathbb{R}^{D_l \times D_l} \tag{29}$$

设 $\boldsymbol{\Sigma}_l$ 的特征值为 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{D_l} \geq 0$。由流形假设和 KL 正则化的维度选择效应，$\boldsymbol{\Sigma}_l$ 的谱呈现**快速衰减**：

$$\lambda_j \begin{cases} = \Theta(1) & \text{if } j \leq d_{\text{eff}} \\ \approx 0 & \text{if } j > d_{\text{eff}} \end{cases} \tag{30}$$

其中 $d_{\text{eff}}$ 为 latent space 的**有效维度**，满足 $d \leq d_{\text{eff}} \leq D_l$。KL 正则化越强（$\beta$-VAE 中 $\beta$ 越大），$d_{\text{eff}}$ 越接近 $d$；KL 正则化越弱，$d_{\text{eff}}$ 可能远大于 $d$（但仍远小于 $D_l$）。

可以通过**累积方差比**来量化：

$$R(k) = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{D_l} \lambda_j} \tag{31}$$

对于典型的图像 VAE（如 SD-VAE），$R(k)$ 在 $k \approx 10^2$ 到 $10^3$ 时已经超过 0.95，而 $D_l = 4096$。这直接给出了 $d_{\text{eff}} / D_l$ 的经验估计。

### 4.3 Jacobian 分析：VAE 如何重塑流形几何

VAE encoder 不仅降低了环境维度（$D \to D_l$），更关键的是改变了流形的**内蕴几何**——曲率、reach、度量张量。我们通过 encoder Jacobian 来分析这一点。

设 encoder 均值映射为 $\boldsymbol{\mu}_\phi: \mathbb{R}^D \to \mathbb{R}^{D_l}$，其 Jacobian 为 $\boldsymbol{J}_\phi(\boldsymbol{x}) \in \mathbb{R}^{D_l \times D}$。流形 $\mathcal{M}$ 上 $\boldsymbol{x}$ 处的切空间 $T_{\boldsymbol{x}}\mathcal{M}$ 经过 encoder 映射到 $\mathcal{M}_l$ 上对应点的切空间：

$$T_{\boldsymbol{x}_l}\mathcal{M}_l = \boldsymbol{J}_\phi(\boldsymbol{x}) \cdot T_{\boldsymbol{x}}\mathcal{M} \tag{32}$$

像素空间中流形 $\mathcal{M}$ 的**黎曼度量张量**（第一基本形式）为 $\boldsymbol{G}(\boldsymbol{x}) = \boldsymbol{J}_\phi^\top \boldsymbol{J}_\phi \in \mathbb{R}^{d \times d}$（限制在切空间上）。编码后流形 $\mathcal{M}_l$ 的度量张量则由 $\boldsymbol{J}_\phi$ 的奇异值决定。

设 $\boldsymbol{J}_\phi$ 在流形切方向上的奇异值为 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_d > 0$。这些奇异值描述了 encoder 对流形不同方向的"拉伸/压缩"程度：

- 如果所有 $\sigma_i \approx c$（某个常数），则 encoder 执行的是近似**等距映射**——流形的内蕴几何被保持
- 如果 $\sigma_1 / \sigma_d \gg 1$（条件数大），则 encoder 对某些方向大幅拉伸、某些方向大幅压缩——流形被扭曲

**VAE 的重建损失隐含地优化了 Jacobian 的条件数。** 原因是：如果 encoder 对某个切方向的奇异值 $\sigma_i$ 很小，则该方向上的信息在 latent code 中几乎丢失，decoder 无法重建。因此，重建损失迫使所有 $\sigma_i$ 保持在合理范围内，从而使 $\boldsymbol{J}_\phi$ 的条件数不至于过大。

### 4.4 流形曲率与 Reach 的改善

流形 $\mathcal{M}$ 在像素空间中的几何通常非常复杂——高曲率、自相交、狭窄的瓶颈区域。这些特征可以用 **reach** $\tau(\mathcal{M})$ 来度量。

**定义 (Reach).** 流形 $\mathcal{M}$ 的 reach 定义为 $\mathcal{M}$ 到其**中轴** (medial axis) 的最小距离：

$$\tau(\mathcal{M}) = \inf_{\boldsymbol{x} \in \mathcal{M}} \sup \left\{ r > 0 : B(\boldsymbol{x}, r) \cap \mathcal{M} \text{ 可被唯一最近点投影} \right\} \tag{33}$$

直觉上，reach 越大，流形越"平坦"，越容易被低维线性子空间局部逼近。Reach 越小，流形越"弯曲"，逼近误差中的常数项越大。

**VAE 的 KL 正则化增大了 latent 流形的 reach。** 数学论证如下：

KL 项将 latent 分布推向各向同性高斯 $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。设 latent code 的边际分布为 $p_l(\boldsymbol{x}_l) = \mathbb{E}_{p_\text{data}}[q_\phi(\boldsymbol{x}_l | \boldsymbol{x})]$。KL 最小化鼓励 $p_l$ 接近 $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。

一个支撑在高曲率流形上的分布不可能接近各向同性高斯——因为高曲率意味着分布集中在弯曲的低维子集上，而非均匀扩散。因此，KL 正则化施加了一个**几何约束**：$\mathcal{M}_l$ 的曲率不能太大。更形式化地，对于满足 KL 约束的 latent 流形：

$$\tau(\mathcal{M}_l) \geq \tau_{\min}(\beta) \tag{34}$$

其中 $\tau_{\min}$ 随 KL 权重 $\beta$ 的增大而增大。这在文献中被称为 VAE 的"后验坍缩"和"解耦"之间的权衡——KL 越强，latent space 越"规整"（reach 更大），但可能丢失细节（重建变差）。

### 4.5 核心定理：VAE 几何预条件化与 x-prediction 的协同

现在我们可以将 VAE 的几何性质与 x-prediction 的优势严格地联系起来。

回顾 3.3 节的逼近定理（式(10)），逼近误差的完整形式（包含常数项）为：

$$\mathcal{E}(N) = O\left( C_s(\mathcal{M}_l)^2 \cdot N^{-2s/d} \right) \tag{35}$$

其中 $C_s(\mathcal{M}_l)$ 是 Hölder 常数，它依赖于流形的几何：

$$C_s(\mathcal{M}_l) \propto \frac{\text{diam}(\mathcal{M}_l)^s}{\tau(\mathcal{M}_l)^s} \cdot \text{vol}(\mathcal{M}_l)^{1/2} \tag{36}$$

其中 $\text{diam}(\mathcal{M}_l)$ 为流形直径，$\text{vol}(\mathcal{M}_l)$ 为流形体积。

**关键观察：** VAE 的 KL 正则化同时作用于这三个量：
- **直径约束**：$\mu_{\phi,j}^2$ 惩罚限制了 $\text{diam}(\mathcal{M}_l)$
- **Reach 增大**：KL 约束增大了 $\tau(\mathcal{M}_l)$（如式(34)）
- **体积控制**：$\sigma_{\phi,j}^2 \to 1$ 的压力使流形不会过度膨胀

这三者的联合效应是**降低 $C_s(\mathcal{M}_l)$**——使得 latent 流形上的函数更容易被神经网络逼近。

现在对比 x-prediction 和 $\epsilon$-prediction 从 VAE 几何中获得的收益：

**x-prediction 的误差：**

$$\mathcal{E}_x(N) = O\left( C_s(\mathcal{M}_l)^2 \cdot N^{-2s/d} \right) \tag{37}$$

VAE 同时降低了常数项 $C_s(\mathcal{M}_l)^2$ 和维度项中的 $d$（通过维度压缩使 $d$ 更"纯净"）。**两个因素都受益于 VAE 的几何优化。**

**$\epsilon$-prediction 的误差：**

$$\mathcal{E}_\epsilon(N) = O\left( N^{-2s/D_l} \right) \tag{38}$$

这里的速率 $N^{-2s/D_l}$ 由**环境维度** $D_l$ 决定，而非流形维度。无论 VAE 如何优化 $\mathcal{M}_l$ 的几何，$D_l$ 是固定的（由 VAE 架构决定）。**VAE 的几何优化对 $\epsilon$-prediction 的逼近速率没有任何帮助。**

这就是协同效应的数学本质：

$$\boxed{\text{VAE 几何优化} \times \text{x-prediction} = C_s(\mathcal{M}_l)^2 \cdot N^{-2s/d} \quad \text{(双重受益)}}$$
$$\text{VAE 几何优化} \times \text{$\epsilon$-prediction} = N^{-2s/D_l} \quad \text{(几何优化被浪费)}$$

### 4.6 Latent Space 的各向异性与 x-prediction 的天然适配

实际的 VAE latent space 并非各向同性的——不同 latent 维度承载的信息量差异巨大。这种**各向异性**可以通过式(29)的协方差矩阵 $\boldsymbol{\Sigma}_l$ 的特征值谱来表征。

设 $\boldsymbol{\Sigma}_l = \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^\top$ 为谱分解，其中 $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_{D_l})$。定义**有效维度** (effective dimensionality)：

$$d_{\text{eff}} = \frac{\left(\sum_{j=1}^{D_l} \lambda_j\right)^2}{\sum_{j=1}^{D_l} \lambda_j^2} = \frac{\text{tr}(\boldsymbol{\Sigma}_l)^2}{\|\boldsymbol{\Sigma}_l\|_F^2} \tag{39}$$

这是 participation ratio 的定义，当所有 $\lambda_j$ 相等时 $d_{\text{eff}} = D_l$（各向同性），当只有一个 $\lambda_j$ 非零时 $d_{\text{eff}} = 1$。

**对于 x-prediction**，网络学习的目标函数自然地沿着 $\boldsymbol{\Sigma}_l$ 的主方向变化——高方差方向对应图像的主要语义变化（姿态、背景、颜色等），低方差方向对应细微纹理或噪声。x-prediction 的损失梯度自动集中在高方差（高信息量）方向上，因为这些方向上的残差更大。

更具体地，考虑 v-loss 下的 x-prediction 损失（式(5)）在 latent 主成分方向上的分解：

$$\mathcal{L}_x = \mathbb{E} \frac{1}{(1-t)^2} \sum_{j=1}^{D_l} (\hat{x}_{\theta,j} - x_{l,j})^2 = \mathbb{E} \frac{1}{(1-t)^2} \sum_{j=1}^{D_l} r_j^2 \tag{40}$$

在训练初期，每个方向的残差 $r_j^2$ 大致正比于该方向的数据方差 $\lambda_j$：

$$\mathbb{E}[r_j^2] \propto \lambda_j \tag{41}$$

因此损失的有效贡献主要来自前 $d_{\text{eff}}$ 个方向，梯度信号天然地与流形的主结构对齐。

**对于 $\epsilon$-prediction**，目标噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ 在所有方向上方差相等（$= 1$）。这意味着每个 latent 维度对损失的贡献相同——无论该维度是否承载有意义的数据信息。网络在 $D_l - d_{\text{eff}}$ 个"无意义"维度上的学习努力，与在 $d_{\text{eff}}$ 个"有意义"维度上的学习努力等量齐观。

这种不匹配可以量化为**有效学习效率比**：

$$\eta_x = \frac{\sum_{j=1}^{d_{\text{eff}}} \lambda_j}{\sum_{j=1}^{D_l} \lambda_j} \approx R(d_{\text{eff}}), \qquad \eta_\epsilon = \frac{d_{\text{eff}}}{D_l} \tag{42}$$

当 latent space 的方差集中度很高时（$R(d_{\text{eff}}) \to 1$ 但 $d_{\text{eff}} / D_l \ll 1$），x-prediction 的学习效率 $\eta_x$ 远高于 $\epsilon$-prediction 的 $\eta_\epsilon$。

### 4.7 小结：VAE 与 x-prediction 的三重协同

综合以上分析，VAE latent space 与 x-prediction 之间存在三重协同效应：

| 协同机制 | VAE 提供的几何性质 | x-prediction 如何利用 | $\epsilon$-prediction 是否受益 |
|---------|------------------|---------------------|:------------------------:|
| **维度压缩** | $D \to D_l$，降低计算成本 | 在更紧凑的空间中执行低维预测 | 同样受益（计算层面） |
| **Reach 增大** | KL 正则展平流形，$\tau(\mathcal{M}_l) \uparrow$ | 降低逼近常数 $C_s$（式(36)） | **不受益**（速率由 $D_l$ 决定） |
| **各向异性对齐** | 数据方差集中在少数方向 | 梯度自动对齐高信息量方向（式(41)） | **不受益**（噪声各向同性） |

第一重协同是所有 latent diffusion 方法共享的。**第二重和第三重是 x-prediction 独占的。** 这解释了为什么 latent x-prediction 的效率增益不仅仅是"不崩溃"——VAE 为 x-prediction 创造了一个几何上优化过的工作空间，而 $\epsilon$/$v$-prediction 无法利用这一优势。

---

## 5. 两步压缩原理

将以上分析综合，latent x-prediction 的优势来自**两步维度压缩的乘性组合**：

$$\underbrace{D \xrightarrow[\text{pixel} \to \text{latent}]{\text{VAE encoder}} D_l}_{\text{第一步：外在维度压缩}} \xrightarrow[\text{噪声} \to \text{干净信号}]{\text{x-prediction}} \underbrace{d}_{\text{第二步：内禀维度利用}}$$

我们可以将现有方法和我们的方法放入一个统一的框架中对比：

| 方法 | 工作空间 | 预测目标维度 | 有效 $D_{\text{eff}}/h$ |
|------|---------|:---------:|:-------------------:|
| Pixel flow matching | $\mathbb{R}^D$ | $D$ (全维噪声/速度) | $D/h \gg 1$ **崩溃** |
| Pixel x-pred (JiT) | $\mathbb{R}^D$ | $d$ (流形) | $d/h < 1$ 但计算量 $\propto D$ |
| Latent $\epsilon$/$v$-pred (DiT) | $\mathbb{R}^{D_l}$ | $D_l$ (全维噪声/速度) | $D_l/h \approx 3\text{–}4$ **隐性低效** |
| **Latent x-pred (ours)** | $\mathbb{R}^{D_l}$ | $d$ (流形) | $d/h < 1$ **最优区间** |

**标准 latent diffusion 只执行了第一步压缩**（$D \to D_l$），获得了 $D/D_l \approx 48\times$ 的计算效率增益。但它没有执行第二步——$\epsilon$/$v$-prediction 仍在 $D_l$ 维空间中工作。

**我们补上了第二步**（$D_l \to d$），在不改变任何计算开销的前提下，获得了额外的 $D_l/d$ 倍统计效率增益。

两步压缩是**乘性**的：总效率增益 $\propto (D/D_l) \times (D_l/d) = D/d$。此前所有方法只走了其中一步。

---

## 6. 为什么这个组合被忽视了？

三个因素共同导致了 latent x-prediction 长期被忽视：

**因素一：JiT 的叙事框架。** JiT 的核心叙事是"自包含的像素空间生成范式"，其全部实验在像素空间进行，自然地将 x-prediction 与"像素空间"绑定。读者的默认推断是"x-prediction 是像素空间的解药"，而非"x-prediction 是一个普适的参数化原则"。

**因素二：不崩溃 ≠ 最优。** JiT 的 Table 2 呈现了一个鲜明的二分法：高维像素空间中 $\epsilon$/$v$-prediction 灾难性失败 (FID 300+)，低维时"差不多" (FID 3–6 的窄幅变化)。在 latent space 中，DiT/SiT 使用 $\epsilon$/$v$-prediction 可以训到 FID 2 以下，结果"看起来没问题"。但"看起来没问题"掩盖了一个事实：**到达相同 FID 可能只需要当前训练步数的几分之一**。

**因素三：Flow matching 的生态惯性。** 在过去两年中，rectified flow / flow matching 成为 latent diffusion 的默认训练范式（SD3、FLUX、SiT）。整个生态——代码库、超参数 recipe、社区经验——都围绕 $v$-prediction 构建。更换预测目标看似微不足道，但需要重新调整噪声调度、CFG 配置、EMA 策略等，这构成了实际的工程惯性。

---

## 7. 初步实验观察

我们将 x-prediction + v-loss 作为即插即用的替换应用于标准 latent diffusion 训练流程：

- 网络架构不变（标准 DiT backbone）
- 噪声调度不变（logit-normal 分布）
- ODE 采样器不变（Euler / Heun）
- **唯一改变**：网络直接输出干净 latent $\hat{\boldsymbol{x}}_\theta$，损失计算时按式(5)转换到 v-space

观察到的现象：

- **拟合效率提升数倍**：达到相同训练损失所需的迭代次数大幅减少
- **收敛曲线明显更陡**：训练早期 loss 下降速度显著快于 $\epsilon$/$v$-prediction baseline
- **零额外开销**：单步计算量、显存占用、参数量完全不变

正如 JiT 论文在 Fig. 7 中展示的：即使使用相同的 v-loss，x-prediction 的训练损失比 $v$-prediction 低约 25%。在像素空间中，这 25% 的 loss 差距最终导致了生成质量的灾难性崩溃（误差在 ODE 多步求解中累积放大）。而在 latent space 中，虽然不会崩溃，但**这 25%（或更多）的 loss 差距直接转化为训练步数的倍数差异**。

详细的定量结果和消融实验将在后续工作中给出。

---

## 8. 讨论与展望

### 7.1 一个通用的参数化原则

本文的分析从 JiT 的流形论证出发，得到了一个更一般的结论：

> **在任何表示空间中训练去噪生成模型时，让网络直接预测干净信号（而非噪声或含噪量）总是更优的参数化选择。**

这个原则的适用范围超出图像生成。蛋白质结构、分子构象、天气数据等高维自然数据同样满足流形假设。无论是否使用 tokenizer 做预处理，x-prediction 都应该是默认选择。

### 7.2 Encoder 与 Predictor 的互补关系

我们的分析揭示了 VAE encoder 和 x-prediction 之间清晰的**分工关系**：

- **Encoder** 负责**计算效率**：将工作空间从 $\mathbb{R}^D$ 压缩到 $\mathbb{R}^{D_l}$，降低前向/反向传播的 FLOPs
- **x-prediction** 负责**统计效率**：将有效学习维度从 $D_l$ 降到 $d$，提高每步梯度和每个参数的信息利用率

此前的 latent diffusion 只使用了 encoder 的计算效率增益，完全没有利用 x-prediction 的统计效率增益。这两种增益是正交的、乘性的。

### 7.3 对 Scaling 的启示

在 diffusion model 的 scaling 浪潮中，训练效率的数倍提升意味着：

- **固定算力下达到更低的 FID**——相当于"免费"的模型质量提升
- **或者，相同质量下用更少的 GPU-hours**——直接降低训练成本

这是一个不改架构、不加参数、零额外计算的"免费午餐"。当 GPU-hours 成为大规模 diffusion 训练的核心瓶颈时，其工程价值不言而喻。

---

## 9. 结语

x-prediction 不是像素空间的特效药，而是流形假设的直接推论——一个与空间选择无关的**通用参数化原则**。将它引入 latent space 是一个"事后看来显然"的组合，却被领域的叙事惯性和"不崩溃即可"的心态所遮蔽。

其数学根基清晰而坚实：
- **逼近论**给出了参数效率的维度依赖性（式(10)–(14)）
- **误差放大分析**揭示了 $\epsilon$-prediction 的固有数值脆弱性（式(16)）
- **梯度信噪比**解释了每步训练的效率差异（式(22)）
- **Fisher 信息矩阵**预测了损失景观条件数的改善（式(24)）

这些分析共同指向一个简洁的结论：

$$\text{Latent Space (计算效率)} \times \text{x-prediction (统计效率)} = \text{最优拟合效率}$$

两步压缩，缺一不可。如果你正在训练 latent diffusion model——**把预测目标从 $\epsilon$ 或 $v$ 换成 $x$**。

---

## 参考文献

1. Li, T. & He, K. (2025). Back to Basics: Let Denoising Generative Models Denoise. *arXiv:2511.13720*.
2. Peebles, W. & Xie, S. (2023). Scalable Diffusion Models with Transformers. *ICCV 2023*.
3. Ma, N. et al. (2024). SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers. *ECCV 2024*.
4. Esser, P. et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *ICML 2024*.
5. Hoogeboom, E. et al. (2023). Simple Diffusion: End-to-End Diffusion for High Resolution Images. *ICML 2023*.
6. Chen, Z. et al. (2025). PixelFlow: Pixel-Space Generative Models with Flow. *arXiv*.
7. Lipman, Y. et al. (2022). Flow Matching for Generative Modeling. *ICLR 2023*.
8. Liu, X. et al. (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR 2023*.
9. Nakada, R. & Imaizumi, M. (2020). Adaptive Approximation and Generalization of Deep Neural Network with Intrinsic Dimensionality. *JMLR*.
10. Chen, M. et al. (2022). Nonparametric Learning on Low-Dimensional Manifolds using Deep ReLU Networks. *JMLR*.
11. Schmidt-Hieber, J. (2019). Nonparametric Regression using Deep Neural Networks with ReLU Activation Function. *Annals of Statistics*.
12. Chapelle, O. et al. (2006). Semi-Supervised Learning. *MIT Press*.
13. Rombach, R. et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.
14. Salimans, T. & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. *ICLR 2022*.
15. Karras, T. et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. *NeurIPS 2022*.
