# Kissing Number（接吻数）问题：算法思路全景 + 我们GA的改进方案（面向“更好下界”）

> 版本：2026-01-08  
> 本文目标：给出“解决/推进 Kissing Number 问题”的主要算法路线（含上界/下界），并针对你们当前 C++ 遗传算法（GA）实现给出可落地的改进建议与工程落地清单。

---

## 0. 问题表述与等价形式（你们现在的建模是对的）

**Kissing number K(d)**：在 d 维欧氏空间中，最多有多少个同半径单位球可以同时与一个中心单位球相切且互不重叠。

常用等价表述（更利于算法）：

### 0.1 球面码 / 角距离约束（你们用的形式）
把外球球心做径向缩放到单位球面 **S^{d-1}**，则每个球心对应单位向量 \(x_i\in\mathbb{R}^d\)，且约束变为

- \(\lVert x_i\rVert = 1\)
- 两两夹角 \(\angle(x_i,x_j)\ge 60^\circ\)
- 等价地：\(\langle x_i,x_j\rangle \le 1/2\)
- 等价地：\(\lVert x_i-x_j\rVert^2 = 2-2\langle x_i,x_j\rangle \ge 1\)

你们代码里用的是 **平方距离 \(\ge 1\)**（`squareDistanceTo > 1` 才合法），是正确的约束形式。

---

## 1. “所有主要算法思路”的地图（按用途分类）

> 备注：Kissing number 的难点在于它同时是**几何 + 非凸全局优化 + 组合结构**问题。主流工作一般分为：  
> - **上界（证明“最多到此为止”）**：LP/SDP 等凸优化/调和分析  
> - **下界（构造/搜索出更大的可行配置）**：格/码构造 + 数值/启发式搜索

### 1.1 上界算法：LP / SDP / 多项式方法（证明用）
**用途**：给定维度 d，证明 \(K(d)\le U\)。  
**典型方法**：
- **Delsarte 线性规划（LP）上界**：用正定核/ Gegenbauer 多项式构造可行证书。
- **Bachoc–Vallentin（Schrijver 风格）SDP 上界**：在 LP 基础上加入三点/更高阶信息，给出更强上界。
- **高精度数值 SDP**：Mittelmann–Vallentin 等实现高精度计算，能给出很多维度最强已知上界。

> 如果你们项目主目标是“找更好解（下界）”，上界不需要自己实现；但报告里强烈建议引用 **Cohn 的 bounds table** 做对比（告诉读者 gap 多大）。

### 1.2 下界算法：结构化构造（“硬核数学”路线）
**用途**：直接构造一个满足约束、点数很大的配置，给出 \(K(d)\ge L\)。

典型结构来源：
- **格（lattice）**：如 Leech lattice（24 维）及其截面/投影构造。
- **误差纠正码 / 常权码**：将二进制码嵌入到球面上，转化为角距离约束。
- **正多胞形/对称群轨道**：利用大对称性制造均匀分布的点。

最近典型进展：
- **Cohn–Li（2024）** 在 17–21 维给出新的下界构造：不是简单 Leech 截面，而是通过“改符号腾空间再加点”的结构化改造获得提升（很值得读一遍，能给你“如何从已知构造再挤点出来”的直觉）。

公开基准资源：
- **Sloane 的 Spherical codes 数据库**：大量“最好已知”配置可以拿来做 seed（初始化/对照）。
- **Henry Cohn 的 kissing number bounds 表**：上下界汇总与引用入口。

### 1.3 下界算法：连续非凸优化（“把它当优化问题”路线）
把问题写成：给定 N，最大化最小距离 \(d_{\min}\)，或最小化约束违背（penalty），在球面流形上优化。

常见技术：
- **投影梯度/黎曼优化（Riemannian optimization）**：每步更新后归一化回单位球面。
- **增广拉格朗日（Augmented Lagrangian）/ 罚函数（Penalty）**：把 \(d_{ij}^2\ge1\) 变成可微惩罚项。
- **多启动（multi-start）**：非凸必备；从不同初值反复跑局部优化。
- **能量法（repulsive potential） + 退火**：先用排斥势把点“抖开”，再修到硬约束可行。

### 1.4 下界算法：随机全局优化/启发式（“工业级拼下界”路线）
典型框架：
- **Simulated Annealing / Basin-Hopping / Tabu / ILS**
- **MLSL（multi-level single linkage）**：全局框架 + 局部求解器
- **VNS（variable neighbourhood search）**
- **粒子群/差分进化/进化策略（CMA-ES 等）**
- **遗传算法 GA / Memetic Algorithm（GA + local search）**

KNP 专门文献里，Kucherenko 等曾用 **MLSL + VNS** 去求 KNP 的数学规划模型（很贴你们“优化”思路）。

### 1.5 下界算法：候选集离散化 + 图/整数规划（“组合优化”路线）
思路：先生成一个大的候选点集 \(C\)，再从中选择最大子集满足两两距离约束。

- 建图：若两点满足约束（距离≥1）则连边（或反过来连“冲突边”）。
- 目标：最大 clique / 最大 independent set / ILP 形式。
- 然后再做连续微调（局部优化）提升 \(d_{\min}\) 或修复边界冲突。

这条路在“你已经有很好的结构候选（来自格/码）”时尤其强。

### 1.6 AI/自动发现（Agent + 搜索）
- DeepMind **AlphaEvolve（2025）**：通过“写代码→评估→进化改代码”找到 11 维新下界 593（从 592 推到 593）。  
  对你们的启发：**把“搜索空间”从点坐标本身扩展到“构造程序/生成规则”**，会大幅提高可迁移性。

---

## 2. 你们当前 GA 实现的关键问题（针对代码的诊断）

你们当前实现的结构（`main.cc` + `PIP.hh`）总结：

- 初始化：全体个体都复制同一个 **±坐标轴** 构造（orthoplex），缺少多样性。
- `fitness()`：若 `isValid()==false` 直接给 0；合法时用“点数 * 1000 + 平均平方距离”的混合指标。
- 交叉：取 parent1 前半点 + parent2 后半点（未做对齐/匹配）。
- 变异：随机替换点（投到球面）+ 以较大概率“直接 push_back 一个随机点”。
- 进化后强行把所有子代点数补到上一代 best 的点数（随机加点）。

这些设计叠加会导致：

### 2.1 “硬判死刑（fitness=0）” + “随机加点”让 N 很难提升
你们最想做的事是从 N 推到 N+1，但随机插入新点几乎必然制造冲突，一旦冲突就被 `fitness=0` 直接淘汰，于是：
- 搜索几乎无法沿“接近可行→可行”的路径前进；
- 强行补点会制造大量非法个体，浪费算力。

### 2.2 目标函数没有对准瓶颈：应以 \(d_{\min}\) 为核心
KNP 的瓶颈永远是最近那一对点。  
“平均距离大”并不意味着“最小距离达标”。尤其在高维，平均项会掩盖局部冲突。

### 2.3 点集交叉需要“旋转对齐 + 匹配”
点集没有天然顺序。你们的“前半/后半拼接”相当于随机拆装结构，难以继承好的几何模式。

### 2.4 工程细节会拖慢（但这是次要问题）
- 每次 `mutate()` 都 `random_device` 重新种子：慢且不可复现；
- `fitness()` 每次都 O(N^2) 全算且不缓存：可优化但先不急。

---

## 3. 对你们算法的改进建议（从“必须改”到“锦上添花”）

下面给一套从易到难、收益很高的改造路线。建议按顺序做，基本每一步都能带来明显提升。

---

### 3.1 必须改 1：把“硬可行判死刑”改成“软约束惩罚 + 修复（Repair）”
把约束
\[
d_{ij}^2 \ge 1
\]
改成可导/可连续的惩罚项，让“差一点可行”的个体也能被保留并朝可行方向推进。

推荐惩罚（hinge-squared）：
\[
\mathrm{penalty}(X)=\sum_{i<j} \max(0, 1-d_{ij}^2)^2
\]

然后把适应度改成（示例）：
- **字典序/双目标（推荐）**：先最大化 N，再最小化 penalty，再最大化 \(d_{\min}\)  
- 或单标量：  
  \[
  \mathrm{score}=M\cdot N - \lambda\cdot \mathrm{penalty} + \epsilon\cdot d_{\min}
  \]
  其中 \(M\) 要远大于 penalty 的典型量级（保证“点数优先”）。

> 好处：你们就可以允许变异产生“暂时不合法”的个体，然后靠 repair/local search 把它们修回来，从而真正实现 N 的增长。

**Repair 思路（任选其一或组合）**：
1. **冲突对驱散**：找最冲突的一对 (i,j)，沿着球面切空间方向把两点推远一点，然后重新归一化；
2. **删点/替换**：对冲突最严重的点进行“重采样”或“局部重插入”；
3. **局部重优化**：对冲突点做几步投影梯度，最小化 penalty。

---

### 3.2 必须改 2：用 \(d_{\min}\) 驱动搜索（而不是平均距离）
建议把“固定 N 的阶段”明确成：**最大化 \(d_{\min}\)**。

推荐运行策略（非常稳）：

**两阶段/分层策略：**
1. 固定 N：优化  
   \[
   \max d_{\min}(X)\quad \text{s.t. }\|x_i\|=1
   \]
   或等价的最小化 penalty。
2. 当 \(d_{\min}\ge1\)（或 penalty≈0）稳定后，再尝试 N→N+1（插入一个新点），继续优化。

这比“可变长直接拼 N”要稳定得多。

---

### 3.3 必须改 3：取消“强行补点到 best 的点数”的粗暴操作
你们现在的补点几乎必把个体搞非法，然后 fitness 归零/penalty 巨大。

替代方案：
- **允许变长个体**：自然选择会保留点数多且 penalty 低的；
- 如果你坚持“最少点数对齐”，那就用**插入启发式**而不是随机点：

**插入启发式（insertion heuristic）**：
- 生成 K 个候选新点（随机/低差异采样/从球面网格取）；
- 选择使 “最小距离最大” 或 “penalty 最小” 的那一个插入；
- 插入后立刻做 repair/local search 若干步。

---

### 3.4 强烈建议：做成 Memetic Algorithm（GA 外面套局部优化）
这是实践里拼下界最常见的强力组合：

**每个 child 产生后：**
1. 做 `repair()`（把主要冲突消掉）
2. 做 10~50 步 `local_optimize()`（投影梯度/坐标下降），最小化 penalty、提升 \(d_{\min}\)
3. 再进入选择

这样 GA 负责跨盆地跳跃，局部优化负责把个体拉回可行/高质量区域。

---

### 3.5 交叉改进：旋转对齐 + 点匹配 + 再交叉（点集问题的关键）
做法：
1. **正交 Procrustes**：找一个旋转矩阵 \(R\) 使得两组点尽量对齐（最小化 \(\sum_i\|x_i-Ry_{\pi(i)}\|^2\)）。
2. **点匹配**：用 Hungarian（或贪心）把两组点一一对应；
3. 对应后再做交叉：  
   - 线性 blend：\(z_i=\mathrm{normalize}(\alpha x_i+(1-\alpha)Ry_i)\)  
   - 或按概率从两边选点，再做 repair

这会让“好结构被继承”，而不是随机拼点。

---

### 3.6 初始化与多样性：别只用一个 orthoplex
建议初始化混合：
- 若干个 **结构 seed**（来自 Sloane 数据库/已知构造）
- 若干个 **随机点集**（用高斯采样再归一化，均匀性更好）
- 若干个 **seed + small jitter + repair**（让结构可探索）

并加上多样性维护（任选其一）：
- Fitness sharing / niching
- 限制近重复个体（例如按 pairwise distance histogram 或 Gram matrix 的粗特征去重）
- 多岛模型（island GA）：多个子种群偶尔迁移

---

### 3.7 速度与可复现（工程优化，推荐做）
- 用单个全局 RNG（固定 seed），避免在 `mutate()` 里反复 `random_device`。
- 缓存最近邻距离：维护每个点的最近邻（或全局最小边），变异只改动少数点时可增量更新。
- OpenMP 并行化 `fitness/penalty` 的双重循环（你们代码结构很适合）。

---

## 4. 推荐你们“第二版”算法框架（可以直接照着实现）

### 4.1 最稳的主框架：Incremental + Memetic
```text
Input: dimension d
X ← initial feasible configuration (seed)
repeat:
  attempt to insert a new point x_new using insertion heuristic
  X ← X ∪ {x_new}
  for t in 1..T:
      X ← repair(X)                  # fix worst violations
      X ← local_optimize(X)          # projected gradient / coordinate descent
  if d_min(X) < 1 (or penalty too big):
      rollback insertion or restart with different x_new
until time budget
output best feasible X found
```

你仍然可以把 GA 用作“生成多个候选插入点/候选配置”的外层随机搜索器。

### 4.2 可选框架：候选集 + 图选择（若你走格/码路线）
```text
C ← generate candidate points from lattice/code construction
Build conflict graph G on C (edge = violates distance constraint)
Solve a maximum independent set / ILP (approx / heuristic)
Refine selected points by local continuous optimization + repair
```

---

## 5. 报告与实验建议（让结果更有说服力）
1. **先复现已知维度**：如 3D 的 12（sanity check）。
2. 用 **Cohn bounds table** 做 baseline：你们找到的 N 属于“已知下界”的哪个档位？gap 多大？
3. 对每个 d，报告：  
   - best N、best \(d_{\min}\)、是否严格可行  
   - 运行时间、重启次数、参数  
4. 如果你们想讨论 Quanta/新进展：可以把 **Cohn–Li(2024)** 的“结构化改造”与 **AlphaEvolve(2025)** 的“自动改算法/构造程序”作为两条很不同但互补的路径。

---

## 6. 参考链接（建议报告引用）
- Quanta Magazine（2025-01-15）关于新进展的科普概览：  
  https://www.quantamagazine.org/mathematicians-discover-new-way-for-spheres-to-kiss-20250115/
- Henry Cohn：kissing number bounds table（强烈建议报告引用）：  
  https://cohn.mit.edu/kissing-numbers/
- Neil Sloane：Spherical codes 数据库（种子/对照）：  
  https://neilsloane.com/packings/
- Bachoc & Vallentin：SDP 上界（经典）：  
  https://arxiv.org/abs/math/0608426
- Mittelmann & Vallentin：高精度 SDP 上界计算：  
  https://arxiv.org/abs/0902.1105
- Kucherenko 等：KNP 数学规划 + MLSL/VNS 搜索：  
  https://www.sciencedirect.com/science/article/pii/S0166218X07000674
- Cohn & Li（2024）：17–21 维改进下界：  
  https://arxiv.org/abs/2411.04916
- DeepMind AlphaEvolve（2025）：11 维下界 593（自动发现）：  
  https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/

---

## 7. 你们下一步“最小改动、最大收益”的 TODO 列表（建议按顺序做）
1. [ ] 把 `fitness=0` 改为 `score = bigM*N - λ*penalty + ε*d_min`（先跑通）
2. [ ] 加 `repair()`：先做“最冲突对驱散/替换”
3. [ ] 取消“强行补点随机塞满”逻辑，改为 insertion heuristic
4. [ ] 每代/每个 child 后加 10~50 步 local optimize（memetic）
5. [ ] 初始化多样化（结构 seed + 随机 + jitter+repair）
6. [ ] 交叉：对齐 + 匹配 后再交叉
7. [ ] 并行化/缓存（提速 + 可复现）

如果你希望我进一步把这些建议“落到你们现有 C++ 代码上”，我可以按 `PIP.hh` 的结构给出一份 **具体的改动补丁（函数签名 + 核心代码）**，让你们直接替换编译运行即可。
