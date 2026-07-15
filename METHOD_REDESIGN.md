# 方法重设计：Graph-Structured Transport Calibration for Stochastic Modular Digital Twins

本文件把讨论中的九节内容落地成一套**自洽、可实现**的方法与实验规范，并说明它与
早期三份脚本（`cascade_koh_flow_experiment.py` / `flow_vs_mcmc_experiment.py` /
`hsgp_discrepancy_experiment.py`）的根本区别。

对应代码：
- `gen_model.py` —— 模型层（随机 GP 模块图 + 祖先采样 + 分解势函数）。
- `graph_transport.py` —— 推断层 Route B（likelihood-tempered SMC + 图结构块 transport）。
- `flow_matching.py` —— 推断层 Route A（amortized graph-structured FMPE + hybrid 校正）。
- `multimodal_testbed.py` —— E8 gate：θ₂ 符号对称多峰 testbed + SMC/VI 双峰验证。
- 保留旧脚本作为 baseline / 消融对照，不再是主线。

---

## 0. 一句话定位

> 把 linked digital twin 建成一个**随机函数组成的概率图模型**，用该图诱导出的
> posterior conditional structure 构造 **graph-structured transport inference**；
> 全程**非线性、无 simulator gradient、完整分布传播、显式利用模块图结构**。

彻底放弃：sensitivity gradient、局部线性化、delta-method、全局 Gaussian moment
matching、"只传播均值方差"。

---

## 1. 与旧路线的根本区别（为什么要推倒重来）

| 维度 | 旧脚本（proof-of-concept） | 本重设计 |
|---|---|---|
| 模块 emulator | 只用 GP predictive **mean**，协方差丢弃 | 保留**完整 GP posterior**，传播 `K_j(V_j,V_j)` 的全 n×n 协方差 |
| 中间状态 `U_j` | 由 `(θ,β)` 确定性推出，不是随机变量 | **显式潜变量**，按模块转移分布随机采样（dgpsi stochastic imputation） |
| discrepancy | 12 维 oracle basis / HSGP（仅系数是潜变量） | 每模块 GP discrepancy（HSGP 白化系数 `c_j`，可换 inducing） |
| 不确定性 | 变分近似 + 单步 IS，几乎只反映 neural approx | 五源不确定性全部由生成模型显式携带（见 §2） |
| 传播方式 | 逐层递推 mean | 祖先随机采样，全局分布可偏态/多峰/重尾 |
| 图结构的使用 | 仅作为 forward 计算图 | 进入 **posterior transport 的条件依赖与稀疏结构**（faithful inverse graph） |
| 求解器 | RealNVP reverse-KL VI；HMC/NeuTra 需 gradient | **梯度自由**的 tempered SMC + 图结构块 MCMC transport |

---

## 2. 模型层：随机函数图（stochastic function graph）

三模块 feed-forward cascade，DAG `G=(V,E)`：
`θ₁,Δ₁ → U₁`，`U₁,θ₂,Δ₂ → U₂`，`U₂,θ₃,Δ₃ → y`；旁支 `U₁→z₁`，`U₂→z₂`。

对 `n` 个 field conditions，把模块 `j` 的所有中间输出写成向量 `U_j ∈ R^n`。**核心**：
给定父状态与参数，模块 GP 给出**联合**条件高斯

```
U_j | U_pa(j), θ_j, c_j
  ~ N( m_j(V_j) + Φ_j(V_j)(√S_j ⊙ c_j),  K_j(V_j,V_j) + σ_ξj² I ),
  V_j = [x, U_pa(j), θ_j].
```

- `(m_j, K_j)`：模块 `j` 的 emulator GP 后验**均值与全协方差**，在 biased simulator
  runs `D_j^sim` 上训练。**保留整块 `K_j(V_j,V_j)`**，不是逐点 marginal variance——
  否则同一 GP 函数在不同 field points 上的相关性消失，posterior uncertainty 被错误压缩。
  （代码：`gen_model.GPModule.predict_mean_cov`。）
- `Δ_j = Φ_j(V_j)(√S_j ⊙ c_j)`：Hilbert-space GP discrepancy，白化系数 `c_j ~ N(0,I)`
  （Solin–Särkkä 2020 的 reduced-rank GP；等价可换成 inducing-variable 表示
  `g_j = Δ_j(Z_j) ~ N(0, K_δ(Z,Z))`）。
- `σ_ξj`：模块 process-noise floor（兼顾数值条件）。
- 因为 `V_2` 含随机的 `U_1`，`K_2` 本身是**逐粒子不同的随机矩阵**——这是"完整协方差"
  的代价，也是与 moment matching 的分界。

**五源不确定性全部显式携带**：emulator uncertainty (`K_j`)、discrepancy uncertainty
(`c_j`)、latent intermediate-state uncertainty (`U_j`)、calibration-parameter
uncertainty (`θ_j`)、measurement noise (`σ_y, σ_z`)。

**潜变量向量**：
```
W = ( θ₁, θ₂, θ₃,  c₁, c₂, c₃,  U₁, U₂ ).
```
`U₁,U₂` 显式（喂给下游非线性 kernel，不可积掉）；终端 `U₃` 解析积掉进 y 似然
（仍保留 emulator-3 协方差），因为没有下游读取它。

**联合分布**（likelihood 与 transition 分离，为 SMC tempering 服务）：
```
p(W) = p(θ) p(c₁)p(c₂)p(c₃) · p(U₁|θ₁,c₁) · p(U₂|U₁,θ₂,c₂)      # prior / transitions
p(O|W) = N(z₁|U₁[I₁],σ_z²) · N(z₂|U₂[I₂],σ_z²) · N(y|m₃(V₃)+Δ₃, K₃+σ_ξ3²+σ_y²)
```
`p(θ_j)=U(-1,1)`。目标 posterior `p(W | y,z₁,z₂, D^sim)`。

> 与 linked GaSP（Kyzyurova 2018；Ming–Guillas 2021；dgpsi Ming–Williamson 2023）的
> 分界：linked GaSP 解析算**经过模块连接后的均值+方差再做 Gaussian moment matching**
> （只对 forward propagation，且 upstream var→0 才精确）。本方法保留模块 GP，但**不在
> 全局层面做 Gaussian moment approximation**，而是随机传播保留完整分布。论文里明确区分：
> *linked GP moment propagation* vs *distributional linked GP posterior propagation*。

---

## 3. 不确定性的"无近似"传播 = 祖先随机采样

"无近似"指：不做 delta method、不做局部线性化、不把全局输出强行近似成 Gaussian。
按模块图祖先采样（`gen_model.StochasticCascade.sample_prior`）：

```
θ⁽ˢ⁾ ~ p(θ);  c_j⁽ˢ⁾ ~ N(0,I)
U₁⁽ˢ⁾ ~ N( m₁+Δ₁, K₁+σ_ξ1² I )                 # 每粒子一次 n×n Cholesky
U₂⁽ˢ⁾ ~ N( m₂(U₁⁽ˢ⁾)+Δ₂, K₂(U₁⁽ˢ⁾)+σ_ξ2² I )
y⁽ˢ⁾ ~ N( m₃(U₂⁽ˢ⁾)+Δ₃, K₃(U₂⁽ˢ⁾)+σ_ξ3²+σ_y² I )
```

每个模块 conditional-on-parents 是高斯，但经非线性模块复合后，`p(y|x,θ)` 通常**非高斯**
（偏态/多峰/重尾）。这正是与原始 linked GaSP 的可观测差别。`graph_transport.predictive_uq`
会报告 `p(y|O)` 在测试点的 skewness / excess kurtosis 作为"分布传播确实重要"的证据。

**关键**：flow / transport 本身**不**负责产生正确的 emulator uncertainty；UQ 来自生成
模型的模块转移。所以论文 claim 是
**full distributional uncertainty propagation through stochastic GP modules**，
而非 "variance propagation"。

---

## 4. Posterior graph ≠ forward DAG 反箭头

forward：`θ₁,c₁ → U₁ → U₂ → y`。观测 `y,z` 后产生 explaining-away 与新依赖：
`θ_j` 与 `c_j`（discrepancy）原本独立，conditioning on collider（`U_j` 的子节点 `y/z`）
后变相关；多父指向同一 collider 也生成新依赖。因此要用 **moralization + variable
elimination** 得到 posterior transport 所需的 separator/Markov-blanket 结构，而不是机械反转。

按模块定义 block `W_j = (θ_j, c_j, U_j)`（`c_j` 即 discrepancy inducing/HSGP 系数）。
在道德图上，各 block 的 **Markov blanket / 需要的势函数**：

| block | 移动的变量 | 触及的势函数（图局部） |
|---|---|---|
| `W₁=(θ₁,c₁,U₁)` | θ₁,c₁ 固定 U₁ / U₁ pCN | `trans₁`（拟合 U₁）；`trans₂`（U₁ 是其输入）；`z₁` |
| `W₂=(θ₂,c₂,U₂)` | θ₂,c₂ 固定 U₂ / U₂ pCN | `trans₂`；`y`（U₂ 输入 emulator-3）；`z₂` |
| `W₃=(θ₃,c₃)` | θ₃,c₃ | `y` |

每个块只需重算**入射到该块**的势函数，不需要看全模型——这就是真正意义上的
graph-structured transport（`graph_transport.block_mh_sweep`），也是模块结构在
**推断层**的价值。三层价值因此齐全：生成结构（局部 GP transition）、不确定性传播
结构（按图逐模块随机传播）、posterior 推断结构（faithful inverse graph 的稀疏 transport）。

---

## 5. 推断层 Route B（已实现）：图结构 tempered SMC

likelihood-tempered 目标序列
```
π_λ(W) ∝ p(W) · p(O|W)^λ,   λ: 0 → 1 （自适应温度）
```
只对 **likelihood** 退火；模块 transition 始终满强度（它们定义 invariant prior）。
`λ=0` 时目标就是精确 prior，用祖先采样精确初始化（生成模型直接作为 SMC 的第 0 步）。

每个温度步：
1. **自适应温度**：二分选 `Δλ` 使 ESS 掉到目标比例（`--ess-target`）。
2. **重加权** `log w += Δλ · log p(O|W)`。
3. ESS 低于阈值时 **systematic resample**。
4. **图结构块 rejuvenation**（Metropolis-within-Gibbs，梯度自由）：
   - 参数块 `(θ_j,c_j)`：小步 RW，潜变量固定 → MH 比只含该块的 transition（+ 终端 `y`）。
   - 潜变量 `U₁,U₂`：**pCN** 移动，天然保持高斯模块转移不变 → MH 比只含下游/观测势
     （`trans₂+z₁` 对 U₁；`z₂+y` 对 U₂）。pCN 是让**紧耦合 GP cascade 真正混合**的关键；
     naïve 各向同性 RW 或"从转移先验整段重抽 U"在紧 y 似然下接受率≈0（已实测踩过坑）。
   - 每块只重算 Markov blanket 内的势函数（§4 表）。

输出带权/重采样 posterior particles `{W_s, w̃_s}`，附 ESS trace、温度数、各块接受率。

> **为什么先做 Route B**：直接瞄准单个真实 dataset 的真 posterior，无 amortization 成本，
> 复用现成 `log_prob` 结构，且给后续 amortized flow 一个**近似精确的参照**去校验。

---

## 6. 推断层 Route A（已实现原型）：amortized graph-structured FMPE

代码：`flow_matching.py`。从完整 cascade 生成 paired samples `(O, θ, c, U) ~ p(·)`
（祖先采样，**无 simulator gradient、训练期不算 likelihood**），训练条件 flow matching
`q_φ(θ,c,U | O) ≈ p(θ,c,U | O)`（FMPE，Dax et al. 2023：CNF 做 SBI，只需 simulator
samples 不需 derivatives）。采用 rectified-flow / linear CFM：`W_t=(1-t)W₀+tW₁`，
回归 `v=W₁-W₀`；推断时从 `N(0,I)` 沿 ODE（Heun）积分到 `t=1`，conditioned on `O_real`。
amortization 范围：**固定 design**（固定 x 网格与 sensor 位置/数量），只对数据实现
amortize，故与 Route B 求解同一 twin，可直接对标并互为校验。

novelty 在于 velocity field 具**图结构**（不是拼大向量喂 generic CNF）：
```
dW_j(t)/dt = v_{φ,j}( W_j(t), W_sep(j)(t), h_j(O), t )
```
- `h_j(O)` 只编码与模块 `j` 相关的观测；
- `W_sep(j)` 传递 faithful inverse graph 上邻接块的 posterior 信息；
- connectivity 由 faithful inverse graph mask；同类型模块可共享参数。

参考：Graphical Normalizing Flows（Wehenkel–Louppe 2021，flow 用指定 BN topology）；
Structured Conditional CNF（从生成图导出 posterior inverse structure，稀疏 ODE amortized）。

**图结构 velocity field**（相对 generic CNF 的 novelty，不是拼大向量喂 flow）：
每个 block `W_j=(rawθ_j,c_j,U_j)` 有独立 velocity head，只看 faithful inverse graph 上
的邻域（道德图诱导的链 `b1–b2–b3`）+ **routed 观测嵌入**（`b1←z1`, `b2←[z2,y]`,
`b3←y`）。connectivity 由图 mask，信息流由概率图决定。

## 6′. hybrid：FM proposal + MC correction（已实现原型）

- **Offline**：GP cascade 祖先模拟训练 `q_φ(W|O)`（无 gradient、传播完整 UQ、用图结构）。
- **Online（本原型）**：真实 `O_real` 下从 `q_φ` 取 particles，作为 §5 图块 MCMC 的初始
  群体，在 `λ=1` 跑若干 rejuvenation sweep 做校正（复用 `graph_transport.block_mh_sweep`，
  **不需要 flow 密度**）。FM 提供快速 proposal；图块 MCMC correction 防 flow collapse、
  把 UQ 拉回真后验。
- **更严格版**：见下 §6″ —— 把 proposal 改成自回归 per-block local flow，`log q_φ`
  逐块解析可得，直接做自归一化 IS `ω_s ∝ p(O_real|W_s)p(W_s)/q_φ(W_s|O_real)` 与 PSIS k̂。

## 6″. Route A′：共享 sequential graph proposal + 三条训练目标（正式并列方法线）

把 cascade posterior inference 读成 multi-turn sequential decision process：**一次
posterior sample = 一个 episode，每个模块 = 一次 sequential inference action**（不是
"每模块一个独立 RL 任务"）。按逆图顺序 π 逐块生成 `W_{π(j)}`，state
`s_j=(O, W_{π(<j)}, G_inv)`，得到自回归联合 proposal

```
q_φ(W|O) = Π_j q_{φ,π(j)}( W_{π(j)} | W_{π(<j)}, O ).
```

**统一观察**：cascade joint 与 proposal 都按模块 factorize，故 per-module local reward

```
r_j = log p(θ_j)+log p(D_j)+log p(U_j|U_pa,θ_j,D_j) [+log p(Z_j|U_j)] − log q_{φ,j}(W_j|s_j)
```

**逐字就是 SMC 第 j 步的 incremental log importance weight**；`Σ_j r_j = log γ(W;O) − log q_φ(W|O)`。
即"RL credit assignment"与"importance weighting"是同一对象——三条线只是训练这个共享
proposal 的三种原理。

**公平性铁律：三条线共用同一个 sequential graph proposal 架构（block W_j=(θ_j,D_j,U_j)，
conditional local flow，逆拓扑序），只换 training objective**；否则差异无法归因。
自回归 local flow ⇒ `log q_φ=Σ_j log q_{φ,j}` 逐块解析，供 IS/PSIS 与 GFlowNet 使用。

| 线 | objective | 性质 | 关键实现注意 |
|---|---|---|---|
| **VI** | `min KL(q_φ‖p)` = entropy-reg RL，terminal reward `log γ` + §3 分解 | on-policy，**mode-seeking** | flow 可 reparameterize ⇒ **pathwise 梯度即可，无需 REINFORCE** |
| **GFlowNet** | Trajectory Balance（固定序 ⇒ backward 平凡）`(log Z_φ+Σlog q_{φ,j}−log γ)²` | off-policy 探索 ⇒ 更 **mode-covering**；fixed point 同为真后验 | 用 **VarGrad / relative-TB** 消掉 `log Z_φ(O)`，避免 amortize Z 的不稳定 |
| **SMC** | learned proposal + 增量权重 + resample + block rejuvenation + **tempering** | **唯一有正确性保证**；给 `log Ẑ_SMC` | proposal 可用 VI/GFlowNet 任一；tempering 兜多峰 |

**判断（重要）**：RL 语言本身不产生新算法——entropy-reg RL = sequential VI；固定序连续
GFlowNet 的 fixed point 也是真后验，其相对 VI 的收益来自 off-policy 探索而非多轨迹。真正
的新颖性在 **graph 结构 + module-level KOH（θ_j–δ_j confounding）+ 部分观测 digital twin**
这个组合，以及"local reward = incremental log-weight"的统一视角。方法主线命名建议
**Graph-Structured Sequential Posterior Sampling for Modular Digital Twins**。

**SMC 救不了 proposal 完全漏掉的峰**（reverse-KL proposal mode-seeking ⇒ 某峰密度≈0 ⇒
权重爆炸/永不采到）；故 learned proposal 必须与从 prior 退火的 tempered SMC 并用。

---

## 7. 与前沿工作的区分（必须清楚划界）

**Deep Gaussian Processes on DAGs（arXiv 2607.09645, 2026-07-10；用户提供，待核）**：
研究 DAG 上 GP 函数组合、中间函数部分观测、uncertainty propagation、保留 graph
dependence 与 collider explaining-away 的 structured VI。其重点是 **DAG deep-GP 函数
学习/emulation + structured VI**。

本方法的重点与之正交/互补：
```
pretrained black-box module emulators
 + module-specific calibration parameters θ_j
 + KOH module discrepancies δ_j
 + physical field observations (partial intermediate z_j)
 + gradient-free graph transport posterior inference
 + Monte Carlo correction
```
尤其 **`θ_j` calibration 与 `δ_j` discrepancy 的 confounding**，不是普通 DAG-DGP
emulation 自动解决的——这是本工作的统计核心（见 §8 的可辨识性主张）。

---

## 8. 实验矩阵（claims → 实验 → 度量）

| # | Claim | 实验 | 主度量 |
|---|---|---|---|
| E1 | 黑盒 KOH 预测 y 好但错校 module θ；保结构+中间观测恢复 θ | 本方法 vs `run_black_box_koh` | θ RMSE、95% coverage、y RMSE |
| E2 | 分布传播≠moment matching | `p(y\|O)` 的 skew/excess-kurtosis；对比只传均值方差的 linked-GaSP moment 版 | 非高斯性、predictive log-score / calibration |
| E3 | 可辨识性随中间观测数变化 | **n_z sweep**（核心图：θ RMSE / coverage vs n_z₁,n_z₂） | 分模块 θ_j RMSE、coverage |
| E4 | 终端模块 θ₃ 最难（discrepancy 落在唯一终端可观测上） | 分模块拆解 E3；`c_δ`(discrepancy 幅度) sweep | θ₃ 相对 θ₁,θ₂ 的 RMSE/coverage 退化 |
| E5 | 图结构 transport 的统计/效率优势 | 图块 SMC vs 单块全局 RW-SMC（同预算） | 温度数、ESS/s、θ RMSE |
| E6 | 求解正确性 | SMC vs（未来）amortized FMPE 交叉校验；SBC rank uniformity | k̂/ESS、SBC KS |
| E7 | emulator UQ 的作用 | 全协方差 vs 仅 mean(σ_ξ=0) 消融 | coverage、predictive log-score |
| **E8** | **多峰 testbed 前置刚需**：单峰下三线等价、主表无故事 | 造真多峰后验（θ 符号/label-switching 或对称 discrepancy）；长跑 tempered SMC 取 ground truth；确认 SMC 找到双峰、单遍 VI 会漏峰 | mode recall、后验双峰可视化 |
| **E9** | **三训练目标并列对比**（VI / GFlowNet / SMC，§6″，共享 proposal 仅换 objective） | 同一 sequential graph proposal 上分别训 reverse-KL VI、VarGrad-TB GFlowNet、learned-proposal tempered SMC | θ RMSE/coverage、**SBC rank uniformity**、mode recall、给 VI/GFlowNet 也算 IS 后 ESS/PSIS k̂、amortized+单次成本 |

统一 synthetic testbed（沿用）：三模块 backbone + true discrepancy，
`θ_true=[0.35,-0.55,0.45]`，`σ_y=0.06, σ_z=0.05`，emulator 用 biased simulator。
多 seed（≥10）报均值±std。**注意：E9 三线对比只有在 E8 多峰 testbed 上才有区分度**——
当前单峰 testbed 上 VI≈GFlowNet≈SMC（已由 FMPE 与充分混合 SMC 的跨方法一致间接印证）。
ground truth 一律用长跑 tempered SMC（Route B 高预算档）。

---

## 9. 方法命名与三层结构

**Graph-Structured Transport Calibration for Stochastic Modular Digital Twins.**

```
模型层     U_j | U_pa(j),θ_j,c_j ~ GP posterior transition（全协方差 + GP discrepancy）
推断层     q(W|O) 由 faithful inverse graph 上 block-structured transport 表示
           (Route B: tempered SMC + 图块 MwG/pCN；Route A: amortized graph-FMPE)
校正层     q → graph-aware tempered SMC → p(W|O)
```
不需要 simulator gradient / 局部线性化 / delta-method / 全局 Gaussian approximation；
真正用到 modular structure 的三层价值（生成 / 传播 / 推断）。

---

## 附 A：运行

```bash
python3 gen_model.py                          # 生成模型 smoke test
python3 graph_transport.py                    # Route B 默认平衡档 512 / n-mh 6 / ess-target 0.5
python3 graph_transport.py --ess-target 0.6 --particles 768 --n-mh 8   # Route B 充分混合参照
python3 flow_matching.py                      # Route A：amortized FMPE + hybrid 校正
# 输出 results/{graph_transport,flow_matching}/{metrics.json, posterior.npz}
```
`--ess-target` 越高 → bridging 分布越多（更慢、混合更充分）；`--particles/--n-mh` 调总预算。
`flow_matching.py` 会加载 `results/graph_transport/posterior.npz`（同 seed）作为 SMC 参照对标。

## 附 B：原型初步结果（validate the pivot）

101 维潜变量（θ 3 + c 38 + U₁,U₂ 各 30；n_field=30，每层 6 个中间 sensor），
梯度自由图结构 SMC vs final-output-only 黑盒 KOH：

| seed | 结构化 θ RMSE | θ 95% coverage | 黑盒 θ RMSE | 黑盒 coverage | `p(y\|O)` 平均 \|skew\| | 温度 / 墙钟 |
|---|---|---|---|---|---|---|
| 7  | **0.067** | [F,T,T] | 0.650 | [T,F,F] | 0.23 | 13 / 13.5s |
| 11 | **0.079** | [T,T,T] | 0.328 | [T,T,F] | 0.32 | 11 / 6.5s |
| 23 | **0.077** | [T,T,T] | 0.398 | [F,F,T] | 0.39 | 12 / 12.9s |

充分混合参照档（seed 7, 768/8/0.6, 425 温度 / ~11min）：θ mean [0.275, **-0.135**, 0.516]、
RMSE 0.246、coverage [T,T,T]、pred y RMSE 0.268、skew 0.125——温度更多 → 后验更平衡、
更宽、更诚实：**θ₂ 被 module-2 discrepancy explaining-away 得更彻底，均值漂到 ≈-0.13**，
而快档 13 温度的"锐 θ₂≈-0.64"是欠混合假象。

### Route A（amortized graph-FMPE）+ hybrid，seed 7

sim 12s + train 126s（一次训练即得 amortized posterior）：

| 方法 | θ mean | θ RMSE | coverage |
|---|---|---|---|
| FMPE `q_φ(W|O)` | [0.349, **-0.154**, 0.461] | 0.229 | [T,T,T] |
| FMPE + 图块 MCMC 校正 | [0.359, -0.191, 0.493] | 0.209 | [T,T,T] |
| Route B SMC 充分混合参照 | [0.275, **-0.135**, 0.516] | 0.246 | [T,T,T] |

**跨方法一致性（E6 的强证据）**：两套完全不同的引擎——梯度自由 tempered SMC 与
amortized flow matching——独立收敛到同一后验，**尤其 θ₂ 的混淆均值高度吻合
（-0.135 vs -0.154）**；hybrid 校正进一步把 θ₁,θ₃ 向真值拉近。这既验证求解正确性，
也确认"θ₂ 宽后验/被 discrepancy 解释掉"是真实统计现象而非某个求解器的 artifact。

**结论**：E1（保结构恢复 θ，远胜黑盒）、E2（`p(y|O)` 显著非高斯 → 分布传播≠moment
matching）、E6（SMC 与 FMPE 跨方法一致）在原型层面均成立。θ₂/θ₃ 的精确可辨识性与
discrepancy explaining-away 强度耦合，正是 §8（n_z / c_δ sweep + SBC）要量化的对象。

### E8 gate 结果（多峰 testbed，已通过 — `multimodal_testbed.py`, seed 7）

module-2 偶函数 `η₂ = 0.65u₁ + cos(1.2πx) + β(θ₂²−c) + γu₁(θ₂²−c)`，β=1.4, γ=0.14；
真值 θ₂=−0.55 ⇒ 后验按对称精确双峰于 ±0.55。

- **(A) tempered graph SMC（1024 粒子, 367 温度, ~410s）→ 覆盖双峰**：θ₂ 两峰
  mean **−0.50 / +0.49**，质量 **0.44 / 0.56 ≈ 50/50**，minority mass 0.44。
  （峰略偏内 ±0.50<0.55：module-2 discrepancy 吸收了 ~0.05，honest。）
- **(B) reverse-KL VI flow（4 restarts）→ 塌峰**：**3/4** restart minority mass = 0.00
  （全塌到 θ₂<0 一侧），1/4 部分塌（0.14）。VI 目标须用 **box-free**（tanh 已保证支撑，
  用 clamp 的 tanh-Jacobian 替代 `theta_in_box` 的 −1e30 硬惩罚，否则 float32 tanh 饱和
  到 ±1 会触发惩罚、梯度爆炸）。

**gate 结论**：mode-covering 的 tempered-SMC 覆盖双峰，mode-seeking 的 reverse-KL VI 塌峰
⇒ E9 三线对比在此 testbed 上有区分度，可以开工。**副发现**：即便后验精确对称，SMC 的
**相对峰质量只近似 50/50（0.44/0.56）**——局部 move 跨不过 θ₂=0 势垒，well-separated 峰的
相对归一化是 SMC 的软肋，正好 motivate E9 里 GFlowNet / mode-jumping 的价值。

**已知的原型级简化**（非 bug，是范围）：CNF 精确 `log q_φ` + IS/PSIS 的严格 hybrid 未做
（现用 particle→MCMC 校正）；FMPE 只在固定 design 上 amortize（未做变 design 的 set-encoder）；
SMC step-size 自适应在温度很少时不充分（提高 `--ess-target`/`--n-mh` 即可）；
离散 GP 超参与 σ_ξ 固定。
