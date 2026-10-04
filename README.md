# rope-embedding

> 零依赖的旋转位置编码（RoPE）实现。从数学定义手写，不引入任何张量库。
> 附正交性、范数保持、相对位置性质的内置数值验证。

```bash
npm run verify
```

```
✔ 正交性 RᵀR = I              最大偏差 2.22e-16
✔ 范数保持 ‖RoPE(x,m)‖ = ‖x‖  最大相对偏差 1.93e-16
✔ 零位置恒等 RoPE(x,0) = x    最大偏差 0.00e+0
✔ 相对位置不变性              平移后最大相对偏差 1.72e-14
✔ 内积只依赖位置差            最大相对偏差 2.01e-16
✔ 可逆性 旋转 −m 可还原        最大偏差 2.22e-16

全部通过
```

`2.22e-16` 是 IEEE 754 双精度的机器 epsilon——误差已经触底，说明实现是精确的，
不是「近似能用」。

## 为什么值得单独拎出来

RoPE 出自 [RoFormer](https://arxiv.org/abs/2104.09864)（Su et al., 2021），
现在是开源大模型的默认选择：LLaMA、InternLM、Qwen、InternVL 都在用。

它的核心价值是一个等式：

```
⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩
```

左边是「两个不同位置的向量各自旋转后做内积」，右边**只依赖位置差 `n−m`**。
也就是说，注意力分数天然只取决于相对位置，不需要额外构造相对位置编码。

这个等式不是近似成立的，是精确成立的——因为旋转是正交变换。
`verifyRoPE()` 会逐条数值验证它。

## 快速开始

零依赖，克隆下来直接跑：

```bash
git clone https://github.com/wulier-arch/rope-embedding.git
cd rope-embedding && npm test     # 48 个用例，无需 npm install
```

用法：

```js
import { applyRoPE, verifyRoPE } from "rope-embedding";

// 形状 [batch, heads, seq, headDim]，最后一维是 headDim，倒数第二维是序列长度
const shape = [1, 8, 16, 64];
const q = new Float64Array(shape.reduce((a, b) => a * b, 1));

const rotated = applyRoPE(q, shape);              // 位置 0..15
const decoded = applyRoPE(q, shape, { offset: 16 }); // 增量解码，位置 16..31
```

不修改入参，返回新的 `Float64Array`。

## ⚠️ 接入已有模型前必读：两种配对约定

数学上只要求「两两配对」，但**配哪两个元素各家实现不一致**，接错会静默出错——
不报错，只是效果变差，极难排查。

| `layout` | 配对方式 | 使用者 |
| --- | --- | --- |
| `'half'`（默认） | `(x_i, x_{i+d/2})` 前后半配对 | LLaMA 系、InternLM、原论文 |
| `'interleaved'` | `(x_{2i}, x_{2i+1})` 相邻配对 | GPT-NeoX 系 |

两者都能满足全部数学性质，只是旋转的维度对不同。`dim=2` 时二者等价，`dim>2` 时结果不同。

**没有默认值能保证正确**——请对着你要接入的模型的源码确认后再选。

## 核心性质

对第 `i` 组（`i` 从 0 开始），旋转角为

```
φ_i = m · θ_i,      θ_i = base^(−2i/d)
```

其中 `d` 是 head 维度，`base` 默认 `10000`。组内做标准二维旋转：

```
x'_{2i}   = x_{2i} · cos φ_i − x_{2i+1} · sin φ_i
x'_{2i+1} = x_{2i} · sin φ_i + x_{2i+1} · cos φ_i
```

由此得到四条性质，全部可数值验证：

| 性质 | 含义 | 为什么重要 |
| --- | --- | --- |
| 正交性 | `R_mᵀ R_m = I` | 旋转不改变内积结构 |
| 范数保持 | `‖RoPE(x,m)‖ = ‖x‖` | 位置编码不污染注意力分数的尺度 |
| 相对位置 | `⟨R_m q, R_n k⟩ = ⟨q, R_{m−n} k⟩` | 注意力天然只看相对距离 |
| 可逆 | 用 `−m` 旋转可还原 | 变换是一一映射，无信息丢失 |

`verifyRoPE()` 就是把这四条（外加零位置恒等、位置差形式）写成可运行的断言：

```js
import { verifyRoPE } from "rope-embedding";

const r = verifyRoPE({ dim: 128, layout: "half" });
if (!r.passed) throw new Error(JSON.stringify(r.checks.filter((c) => !c.passed), null, 2));
```

设计思路参考了 [axon](https://github.com/wulier-arch/axon) 的 `checkGradient`——
不靠「跑起来没报错」，而是用数学上必须成立的性质去对照。

## API

| 函数 | 用途 |
| --- | --- |
| `applyRoPE(data, shape, opts?)` | 对最后一维应用 RoPE，返回新的 `Float64Array` |
| `applyRoPEToVector(vec, position, opts?)` | 只旋转一个向量 |
| `ropeFrequencies(dim, opts?)` | 频率表 `θ_i`，长度 `dim/2` |
| `ropeMatrix(position, dim, opts?)` | 显式旋转矩阵 `dim×dim`（用于验证与教学） |
| `ropeInnerProduct(q, k, m, n, opts?)` | `⟨RoPE(q,m), RoPE(k,n)⟩` |
| `verifyRoPE(opts?)` | 内置性质自检 |

### `opts`

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `base` | `10000` | 频率基数 |
| `layout` | `'half'` | `'half'` \| `'interleaved'` |
| `offset` | `0` | 位置起点。增量解码时传已生成的 token 数 |
| `positions` | `null` | 显式指定每个位置，长度须等于 `seq` |
| `scaling` | `null` | 上下文扩展配置，见下 |

`shape` 支持 `[seq, dim]`、`[heads, seq, dim]`、`[batch, heads, seq, dim]`——
最后一维是 `headDim`，倒数第二维是序列长度。

## 上下文扩展

长上下文外推用两种经典策略，都已实现：

```js
// linear：位置除以 factor，最简单，训练后微调效果最好
applyRoPE(q, shape, { scaling: { type: "linear", factor: 4 } });

// ntk：放大 base，低频分量变化更慢，等效拉长可分辨的位置范围
applyRoPE(q, shape, { scaling: { type: "ntk", factor: 4 } });
```

`ntk` 的指数取 `d/(d−2)`，是 NTK-aware 插值的经典形式。两者结果不同，选哪个取决于
你的模型是怎么训练的——**推理时的缩放策略必须与训练时一致**，否则位置语义会错位。

## 增量解码

KV cache 场景下，每步只需要旋转新来的 token，用 `offset` 指定它的位置：

```js
// 第 1 步：prefill，位置 0..15
const k1 = applyRoPE(kCache, [1, 8, 16, 64], { offset: 0 });

// 第 2 步：decode，新 token 的位置是 16
const k2 = applyRoPE(newToken, [1, 8, 1, 64], { offset: 16 });
```

测试里有专门一条用例验证「分段旋转等价于一次性旋转整个序列」——
这是 KV cache 正确性的关键，写错会导致长序列下位置错乱。

## 已知边界

- **只支持偶数 `headDim`**。RoPE 按两两配对定义，奇数维度无定义，会直接抛错。
- **不处理 batch 维度的位置差异**。同一次调用里所有 batch 共用同一套位置，
  这是标准做法；如果你的场景需要 per-batch 位置，请拆成多次调用。
- **`dim=2` 时两种 layout 等价**，写测试时注意别用它来区分布局。
- **精度**：使用 `Math.cos` / `Math.sin`，在位置极大（> 1e6）时角度精度会下降。
  常规上下文长度下误差在机器 epsilon 量级。

## 与 axon 的关系

[axon](https://github.com/wulier-arch/axon) 是一个零依赖神经网络框架，已经有
`MultiHeadAttention` 和 `TransformerBlock`——但**没有位置编码**。

这个库补上了缺的那块。两者都遵循同一套做法：零依赖、手写、把正确性摊开给你看。

## 出处与许可

算法出自 Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*
([arXiv:2104.09864](https://arxiv.org/abs/2104.09864))。

本项目是**独立实现**——依据论文的数学定义编写，不复制任何现有代码库。
实现过程中参考了 LLaMA 与 InternLM 的公开文档以确认配对约定，两者均为
Apache-2.0 / 各自许可。

[MIT](LICENSE) © 2026 wulier-arch
