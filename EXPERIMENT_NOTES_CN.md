# Linked Digital Twin Cascade KOH 实验说明

本文档记录当前实验的 ground truth、digital twin emulator、黑盒 KOH baseline，以及 structured flow posterior inference 的具体设定。对应代码文件是 `cascade_koh_flow_experiment.py`。

## 1. 实验目标

这个实验想验证一个核心现象：

```text
只看最终输出的黑盒 KOH calibration 可能把 y 预测得很好，
但 module-level calibration parameter 会严重错位；

如果保留 linked digital twin 的模块结构，并加入少量中间观测，
则可以显著改善每个模块 theta_j 的恢复。
```

因此实验不是为了证明 flow 一定比 MCMC 准，而是为了构造一个可控 synthetic setting，展示：

- known modular structure 对 calibration 有价值；
- final-output discrepancy 会吸收参数误差；
- sparse intermediate observations 可以缓解 module-level confounding；
- flow 可以作为复杂 structured posterior 的 flexible approximation。

## 2. Ground Truth Physical System

真实物理系统是一个三模块 feed-forward cascade：

```text
x, theta_1 -> u_1
x, u_1, theta_2 -> u_2
x, u_2, theta_3 -> y
```

真实 calibration parameter 是：

```text
theta_true = [0.35, -0.55, 0.45]
```

真实系统由 simulator backbone 加真实 discrepancy 组成：

```text
u1 = eta1(x, theta1) + delta1_true(x)
u2 = eta2(x, u1, theta2) + delta2_true(x, u1)
y  = eta3(x, u2, theta3) + delta3_true(x, u2)
```

其中 simulator backbone 是：

```text
eta1(x, theta1)
= sin(1.6 pi x) + 0.75 theta1 + 0.15 x theta1

eta2(x, u1, theta2)
= 0.65 u1 + cos(1.2 pi x + 0.35 theta2)
  + 0.45 theta2 + 0.08 u1 theta2

eta3(x, u2, theta3)
= 0.80 u2 + sin(0.75 u2 + theta3)
  + 0.25 theta3 + 0.10 x
```

真实 discrepancy 是：

```text
delta1_true(x)
= 0.22 cos(2.2 pi x) + 0.08(x - 0.5)

delta2_true(x, u1)
= 0.16 sin(1.7 pi x + 0.6 u1) + 0.045(u1^2 - 0.7)

delta3_true(x, u2)
= -0.18 cos(1.4 pi x - 0.35 u2) + 0.06 sin(3 pi x)
```

注意：field data 来自 `eta + true discrepancy`，不是来自 emulator。这一点刻意制造了 digital twin bias。

## 3. Digital Twin Emulator

digital twin 不知道真实 discrepancy。每个子模块只用 biased simulator `eta_j` 的 simulation runs 训练 GP emulator：

```text
GP1 learns eta1(x, theta1)
GP2 learns eta2(x, u1, theta2)
GP3 learns eta3(x, u2, theta3)
```

每个模块默认使用：

```text
n_sim = 70
```

训练输入分别是：

```text
module 1 input: (x, theta1)
module 2 input: (x, u1, theta2)
module 3 input: (x, u2, theta3)
```

GP emulator 使用手写 RBF kernel regression：

```text
k(a, b) = exp(-0.5 ||(a - b) / lengthscale||^2)
```

当前实现只使用 GP predictive mean，没有显式传播 GP predictive variance。linked emulator 递推为：

```text
u1_hat = GP1(x, theta1)
u2_hat = GP2(x, u1_hat, theta2)
y_hat  = GP3(x, u2_hat, theta3)
```

因此当前实现是 linked GP 思路的简化版：保留模块结构，但暂时不传播 emulator uncertainty。

## 4. Field Observations

默认生成：

```text
n_field = 40
```

所有样本都观测最终输出：

```text
y_obs = y_true + N(0, sigma_y^2)
sigma_y = 0.06
```

同时只观测少量中间状态：

```text
z1_obs = u1_true + N(0, sigma_z^2)
z2_obs = u2_true + N(0, sigma_z^2)
sigma_z = 0.05
```

默认每层中间状态只观测：

```text
n_z1 = 8
n_z2 = 8
```

所以这是一个 partially observed cascade calibration 问题：大多数中间状态是 latent 的，只有少量 noisy sensor observations。

## 5. Baseline: 黑盒 KOH Calibration

黑盒 baseline 不使用中间观测，也不显式建模 module discrepancy。它只看最终输出：

```text
y_obs = linked_emulator(x, theta) + delta_y(x) + epsilon
```

其中 final discrepancy 是：

```text
delta_y ~ GP(0, K_delta)
epsilon ~ N(0, sigma_y^2 I)
```

对固定 `theta`，`delta_y` 可以被 Gaussian marginalization 积分掉：

```text
y_obs | theta
~ N(mu_y(theta), K_delta + sigma_y^2 I)
```

其中：

```text
mu_y(theta) = GP3(x, GP2(x, GP1(x, theta1), theta2), theta3)
```

log posterior 是：

```text
log p(theta | y)
= -0.5 (y - mu_y(theta))^T
      (K_delta + sigma_y^2 I)^(-1)
      (y - mu_y(theta))
  + log p(theta)
```

prior 是：

```text
theta_j ~ Uniform(-1, 1)
```

采样方法是 random-walk Metropolis。

这个 baseline 的特点是 final discrepancy 非常灵活，所以它可以把最终 `y` 拟合得很好；但它也很容易把 module-level 参数校错，因为 downstream discrepancy 可以吸收 upstream 参数误差。

## 6. Structured Cascade KOH Model

structured 方法保留模块结构，并给每个模块一个低秩 discrepancy basis：

```text
u1 = GP1(x, theta1) + B1(x)^T alpha1

u2 = GP2(x, u1, theta2) + B2(x, u1)^T alpha2

y  = GP3(x, u2, theta3) + B3(x, u2)^T alpha3
```

basis 设定为：

```text
B1(x)
= [1, sin(2 pi x), cos(2 pi x), x - 0.5]

B2(x, u1)
= [1, sin(pi x), u1, u1^2 - 0.7]

B3(x, u2)
= [1, cos(pi x), sin(u2), x u2]
```

每个模块 4 个 discrepancy coefficients，总共 12 个。代码中使用：

```text
alpha_j = alpha_scale * beta_j
beta_j ~ N(0, I)
alpha_scale = 0.18
```

所以 structured posterior 的未知量是：

```text
theta = (theta1, theta2, theta3)
beta  = (beta1, ..., beta12)
```

似然包含最终输出和 sparse intermediate observations：

```text
y_obs  ~ N(y_hat(theta, beta), sigma_y^2 I)
z1_obs ~ N(u1_hat(theta, beta), sigma_z^2 I)
z2_obs ~ N(u2_hat(theta, beta), sigma_z^2 I)
```

目标 posterior 是：

```text
p(theta, beta | y_obs, z1_obs, z2_obs)
propto
p(y_obs | theta, beta)
p(z1_obs | theta, beta)
p(z2_obs | theta, beta)
p(beta)
p(theta)
```

## 7. Flow Posterior Inference

flow 使用 RealNVP normalizing flow。被 flow 表示的是一个 15 维无约束变量：

```text
w = [raw_theta1, raw_theta2, raw_theta3,
     beta1, ..., beta12]
```

其中：

```text
theta = tanh(raw_theta)
```

这样可以保证：

```text
theta_j in (-1, 1)
```

RealNVP 定义一个可逆变换：

```text
z ~ N(0, I)
w = T_psi(z)
```

因此可以计算 variational density：

```text
q_psi(w)
```

训练目标是反向 KL：

```text
min_psi KL(q_psi(w) || p(w | data))
```

实现中每一步：

```text
w_sample, logq = flow.sample_and_logq(batch_size)
logp = target.log_prob(w_sample)
loss = mean(logq - logp)
```

这等价于最大化 ELBO：

```text
ELBO = E_q[log p(w, data) - log q(w)]
```

训练完成后，从 flow 中采样：

```text
w_s ~ q_psi(w)
theta_s = tanh(raw_theta_s)
beta_s = beta_s
```

然后做 self-normalized importance correction：

```text
weight_s ∝ p(w_s | data) / q_psi(w_s)
```

代码中使用未归一化 posterior：

```text
logw = logp - logq
```

并报告 effective sample size：

```text
ESS = 1 / sum(weight_s^2)
```

默认运行中：

```text
ESS = 1264 / 12000
```

这说明 flow proposal 不完美，但已经能覆盖目标 posterior 的主要质量区域。

## 8. 默认实验结果

默认命令：

```bash
python3 cascade_koh_flow_experiment.py
```

输出路径：

```text
results/cascade_koh_flow/metrics.json
results/cascade_koh_flow/posterior_samples.npz
```

默认结果：

```text
true theta = [0.35, -0.55, 0.45]

black-box KOH theta mean = [0.437, 0.421, -0.654]
black-box theta RMSE     = 0.850
black-box y RMSE         = 0.051

structured flow + IS theta mean = [0.319, -0.466, 0.547]
structured flow theta RMSE      = 0.076
structured flow y RMSE          = 0.097
structured flow ESS             = 1264 / 12000
```

解释：

```text
black-box KOH 的 final discrepancy 很灵活，
所以最终 y prediction 很好；
但它牺牲了 module-level theta identifiability。

structured cascade KOH 使用模块结构和少量中间观测，
因此 theta 恢复明显更好；
但由于 discrepancy 被限制为低秩 module basis，
y RMSE 不如带自由 final GP discrepancy 的 black-box baseline。
```

## 9. 当前实现的简化与后续扩展

当前版本是 proof-of-concept，有几个重要简化：

1. GP emulator 只使用 predictive mean，没有传播 predictive variance。
2. module discrepancy 是低秩 basis，不是完整 GP discrepancy。
3. flow 是针对单个 dataset 的 posterior approximation，不是 amortized posterior estimator。
4. 中间 latent states `U` 没有作为独立变量采样，而是由 `(theta, beta)` 通过 linked forward model deterministic 地推出。
5. black-box KOH 使用 final GP discrepancy，因此在 `y RMSE` 上有天然优势。

可以进一步扩展为：

- 给 structured model 也加入 final residual discrepancy；
- 把 module discrepancy 从 basis 扩展为 sparse GP / inducing-point GP；
- 显式传播 module GP emulator uncertainty；
- 把 `U` 作为 latent variable 放入 posterior；
- 用 SMC / MCMC correction 替代简单 importance correction；
- 做多 seed、多 n_field、多 intermediate observation rate 的系统实验；
- 加 simulation-based calibration 和 posterior predictive check。
