/**
 * 旋转位置编码（Rotary Position Embedding, RoPE）。
 *
 * 出处：Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
 *       (arXiv:2104.09864)。现代开源大模型基本都在用：LLaMA、InternLM、Qwen、InternVL…
 *
 * 这个文件只做一件事：把 RoPE 的数学定义写成可读、可验证的代码。
 * 零依赖，不引入任何张量库——输入输出都是扁平的 Float64Array 加一个 shape。
 *
 * ── 核心思想 ──────────────────────────────────────────────────────────────
 *
 * 绝对位置编码（正弦波、可学习 embedding）是把位置信息「加」到输入上。
 * RoPE 换了个思路：把向量按维度两两分组，每组看成一个二维平面上的点，
 * 然后用位置 m 决定的角度去「旋转」它。
 *
 * 对第 i 组（i 从 0 开始），旋转角为
 *
 *     φ_i = m · θ_i,      θ_i = base^(−2i/d)
 *
 * 其中 d 是 head 维度，base 默认 10000。分组内做标准二维旋转：
 *
 *     x'_{2i}   = x_{2i} · cos φ_i − x_{2i+1} · sin φ_i
 *     x'_{2i+1} = x_{2i} · sin φ_i + x_{2i+1} · cos φ_i
 *
 * ── 为什么这样设计 ────────────────────────────────────────────────────────
 *
 * 旋转是正交变换，所以 ‖RoPE(x, m)‖ = ‖x‖ —— 位置编码不会改变向量的模长，
 * 不会污染注意力分数的尺度。
 *
 * 更关键的是内积性质。设 R_m 是位置 m 的旋转算子，则
 *
 *     ⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩
 *
 * 左边是「两个不同位置的向量各自旋转后做内积」，右边只依赖位置差 n−m。
 * 也就是说：**注意力分数天然只取决于相对位置**，不需要显式构造相对位置编码。
 * 这个等式就是 RoPE 的全部价值所在，`verifyRoPE()` 会逐条数值验证它。
 *
 * ── 两种分组约定 ──────────────────────────────────────────────────────────
 *
 * 数学上只要求「两两配对」，但配哪两个元素各家实现不一致，接错会静默出错：
 *
 *   'half'        （前后半配对）(x_i, x_{i+d/2})   LLaMA 系、InternLM、原论文
 *   'interleaved' （相邻配对）  (x_{2i}, x_{2i+1})  GPT-NeoX 系
 *
 * 两者都能满足上面全部性质，只是旋转的维度对不同。**接入已有模型前务必确认
 * 对方用的是哪种**——这是最常见的 RoPE 集成 bug，而且不会报错，只会让效果变差。
 */

/** 支持的配对约定 */
export const LAYOUTS = Object.freeze(["half", "interleaved"]);

/** 默认频率基数。原论文与 LLaMA 都用 10000 */
export const DEFAULT_BASE = 10000;

/* ------------------------------------------------------------------ 内部工具 */

function assertShape(shape) {
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new Error("shape 必须是非空数组，例如 [batch, heads, seq, headDim]");
  }
  for (const s of shape) {
    if (!Number.isInteger(s) || s <= 0) {
      throw new Error(`shape 的每一维都必须是正整数，得到 [${shape}]`);
    }
  }
}

function numElements(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function assertLayout(layout) {
  if (!LAYOUTS.includes(layout)) {
    throw new Error(`layout 必须是 ${LAYOUTS.join(" 或 ")}，得到 ${JSON.stringify(layout)}`);
  }
}

/**
 * 把上下文扩展配置折算成实际的频率基数。
 *
 * 'linear' 不动 base，而是把位置除以 factor（见 resolvePosition）。
 * 'ntk'    不动位置，而是把 base 放大——低频分量变化更慢，等效于拉长可分辨的
 *          位置范围。指数取 d/(d−2) 是 NTK-aware 插值的经典形式。
 */
function resolveBase(dim, base, scaling) {
  if (!scaling) return base;
  if (scaling.type === "ntk") {
    const { factor } = scaling;
    if (!(factor > 1)) throw new Error(`ntk 缩放的 factor 必须大于 1，得到 ${factor}`);
    return base * Math.pow(factor, dim / (dim - 2));
  }
  if (scaling.type === "linear") return base;
  throw new Error(`未知的 scaling.type：${JSON.stringify(scaling.type)}（支持 'linear' 与 'ntk'）`);
}

/** 把位置折算成实际参与旋转的「有效位置」 */
function resolvePosition(position, scaling) {
  if (scaling && scaling.type === "linear") return position / scaling.factor;
  return position;
}

/* ------------------------------------------------------------------ 公开 API */

/**
 * 计算各分组的旋转频率 θ_i = base^(−2i/d)。
 *
 * 返回值长度为 d/2。θ 随 i 指数衰减：第 0 组转得最快（波长最短），
 * 最后一组转得最慢（波长最长，负责长距离区分）。
 *
 * @param {number} dim  head 维度，必须为正偶数
 * @param {object} [opts]
 * @param {number} [opts.base=10000]
 * @param {{type:'linear'|'ntk', factor:number}} [opts.scaling] 上下文扩展配置
 * @returns {Float64Array} 长度 dim/2
 */
export function ropeFrequencies(dim, opts = {}) {
  const { base = DEFAULT_BASE, scaling = null } = opts;
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new Error(`dim 必须是正整数，得到 ${dim}`);
  }
  if (dim % 2 !== 0) {
    throw new Error(`dim 必须是偶数（两两配对），得到 ${dim}`);
  }
  const half = dim / 2;
  const b = resolveBase(dim, base, scaling);
  const out = new Float64Array(half);
  for (let i = 0; i < half; i++) {
    out[i] = Math.pow(b, (-2 * i) / dim);
  }
  return out;
}

/**
 * 生成位置 `position` 对应的旋转矩阵（dim × dim，行主序）。
 *
 * 它是块对角的：沿对角线排列 dim/2 个 2×2 旋转块。主要用于验证——
 * 正交性、范数保持这些性质用矩阵表达最直观。
 *
 * @param {number} position
 * @param {number} dim
 * @param {object} [opts] 同 ropeFrequencies
 * @returns {Float64Array} 长度 dim*dim
 */
export function ropeMatrix(position, dim, opts = {}) {
  const { layout = "half", scaling = null } = opts;
  assertLayout(layout);
  const freq = ropeFrequencies(dim, opts);
  const half = dim / 2;
  const phi = resolvePosition(position, scaling);
  const m = new Float64Array(dim * dim);
  for (let i = 0; i < half; i++) {
    const angle = phi * freq[i];
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const [a, b] = layout === "half" ? [i, i + half] : [2 * i, 2 * i + 1];
    m[a * dim + a] = c;
    m[a * dim + b] = -s;
    m[b * dim + a] = s;
    m[b * dim + b] = c;
  }
  return m;
}

/**
 * 对张量的最后一维应用 RoPE。
 *
 * 输入是扁平数据加 shape，shape 的最后一维是 headDim，倒数第二维是序列长度：
 *
 *     [seq, dim]              → 单头
 *     [heads, seq, dim]       → 多头
 *     [batch, heads, seq, dim] → 完整注意力输入
 *
 * 不修改入参，返回新的 Float64Array。
 *
 * @param {Float64Array|number[]} data
 * @param {number[]} shape
 * @param {object} [opts]
 * @param {number}   [opts.base=10000]
 * @param {number}   [opts.offset=0]      位置起点，增量解码时传已生成的 token 数
 * @param {string}   [opts.layout='half'] 'half' | 'interleaved'
 * @param {number[]} [opts.positions]     显式指定每个位置，长度须等于 seq
 * @param {object}   [opts.scaling]       上下文扩展，见 ropeFrequencies
 * @returns {Float64Array}
 */
export function applyRoPE(data, shape, opts = {}) {
  const {
    base = DEFAULT_BASE,
    offset = 0,
    layout = "half",
    positions = null,
    scaling = null,
  } = opts;

  assertShape(shape);
  assertLayout(layout);

  const dim = shape[shape.length - 1];
  const seq = shape.length >= 2 ? shape[shape.length - 2] : 1;
  const total = numElements(shape);

  if (data.length !== total) {
    throw new Error(`data 长度 ${data.length} 与 shape [${shape}] 的元素数 ${total} 不符`);
  }
  if (!Number.isInteger(offset)) {
    throw new Error(`offset 必须是整数，得到 ${offset}`);
  }
  if (positions !== null) {
    if (!Array.isArray(positions) || positions.length !== seq) {
      throw new Error(`positions 长度必须是 ${seq}（序列长度），得到 ${positions?.length}`);
    }
  }

  const freq = ropeFrequencies(dim, { base, scaling });
  const half = dim / 2;
  const out = new Float64Array(data);
  const nVec = total / dim;

  // 预先算好每个序列位置的角度，避免在向量循环里重复求三角函数
  const cosTable = new Float64Array(seq * half);
  const sinTable = new Float64Array(seq * half);
  for (let s = 0; s < seq; s++) {
    const raw = positions ? positions[s] : s + offset;
    const phi = resolvePosition(raw, scaling);
    for (let i = 0; i < half; i++) {
      const angle = phi * freq[i];
      cosTable[s * half + i] = Math.cos(angle);
      sinTable[s * half + i] = Math.sin(angle);
    }
  }

  for (let v = 0; v < nVec; v++) {
    const s = seq > 1 ? v % seq : 0;
    const off = v * dim;
    const tOff = s * half;
    for (let i = 0; i < half; i++) {
      const c = cosTable[tOff + i];
      const sn = sinTable[tOff + i];
      const [a, b] = layout === "half" ? [i, i + half] : [2 * i, 2 * i + 1];
      const x = out[off + a];
      const y = out[off + b];
      out[off + a] = x * c - y * sn;
      out[off + b] = x * sn + y * c;
    }
  }

  return out;
}

/**
 * 只旋转一个向量。便捷入口，等价于 shape 为 [1, dim] 的 applyRoPE。
 *
 * @param {Float64Array|number[]} vec
 * @param {number} position
 * @param {object} [opts] 同 applyRoPE（positions 会被忽略）
 * @returns {Float64Array}
 */
export function applyRoPEToVector(vec, position, opts = {}) {
  const dim = vec.length;
  // shape 是 [1, dim]，seq = 1，所以显式给一个长度为 1 的 positions 即可表达任意位置
  const { positions: _dropPositions, offset: _dropOffset, ...rest } = opts;
  return applyRoPE(new Float64Array(vec), [1, dim], { ...rest, positions: [position] });
}

/**
 * 内积验证辅助：返回 ⟨RoPE(q, m), RoPE(k, n)⟩。
 *
 * 单独提供这个函数，是因为「只依赖相对位置」是 RoPE 的核心性质，
 * 而直接写出来最容易看懂。
 */
export function ropeInnerProduct(q, k, m, n, opts = {}) {
  if (q.length !== k.length) {
    throw new Error(`q 与 k 维度必须相同，得到 ${q.length} 与 ${k.length}`);
  }
  const rq = applyRoPEToVector(q, m, opts);
  const rk = applyRoPEToVector(k, n, opts);
  let acc = 0;
  for (let i = 0; i < rq.length; i++) acc += rq[i] * rk[i];
  return acc;
}

/* ------------------------------------------------------------------ 性质自检 */

/** 确定性 PRNG（mulberry32），保证自检结果可复现 */
function makeRng(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomVec(n, rng) {
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = rng() * 2 - 1;
  return v;
}

function norm(v) {
  let acc = 0;
  for (let i = 0; i < v.length; i++) acc += v[i] * v[i];
  return Math.sqrt(acc);
}

/**
 * 内置性质自检：把「RoPE 为什么是对的」变成可运行的断言。
 *
 * 设计思路与 axon 的 checkGradient 一致——不靠「跑起来没报错」，而是用
 * 数学上必须成立的性质去对照。任何一处实现错误都会让对应条目失败。
 *
 * 检查项：
 *   1. 正交性      R_mᵀ R_m = I
 *   2. 范数保持    ‖RoPE(x, m)‖ = ‖x‖
 *   3. 零位置恒等  RoPE(x, 0) = x
 *   4. 相对位置不变 ⟨R_m q, R_n k⟩ = ⟨R_{m+t} q, R_{n+t} k⟩
 *   5. 位置差形式  ⟨R_m q, R_n k⟩ = ⟨R_{m−n} q, k⟩
 *   6. 可逆性      用 −m 旋转可还原
 *
 * @param {object} [opts]
 * @param {number} [opts.dim=64]
 * @param {number} [opts.base=10000]
 * @param {string} [opts.layout='half']
 * @param {object} [opts.scaling=null]
 * @param {number} [opts.tolerance=1e-9]
 * @param {number} [opts.seed=42]
 * @returns {{passed:boolean, checks:Array<{name:string,passed:boolean,detail:string}>}}
 */
export function verifyRoPE(opts = {}) {
  const {
    dim = 64,
    base = DEFAULT_BASE,
    layout = "half",
    scaling = null,
    tolerance = 1e-9,
    seed = 42,
  } = opts;

  assertLayout(layout);
  const rng = makeRng(seed);
  const checks = [];
  const add = (name, passed, detail) => checks.push({ name, passed, detail });
  const pass = (name, detail) => add(name, true, detail);
  const fail = (name, detail) => add(name, false, detail);
  const exp = (x) => x.toExponential(2);

  const positions = [0, 1, 7, 128, 1000, 8191];

  /* 1. 正交性 */
  {
    let worst = 0;
    for (const m of positions) {
      const R = ropeMatrix(m, dim, { base, layout, scaling });
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          let acc = 0;
          for (let k = 0; k < dim; k++) acc += R[k * dim + i] * R[k * dim + j];
          worst = Math.max(worst, Math.abs(acc - (i === j ? 1 : 0)));
        }
      }
    }
    worst <= tolerance
      ? pass("正交性 RᵀR = I", `最大偏差 ${exp(worst)}`)
      : fail("正交性 RᵀR = I", `最大偏差 ${exp(worst)} 超过容差 ${tolerance}`);
  }

  /* 2. 范数保持 */
  {
    let worst = 0;
    const x = randomVec(dim, rng);
    const n0 = norm(x);
    for (const m of positions) {
      const n1 = norm(applyRoPEToVector(x, m, { base, layout, scaling }));
      worst = Math.max(worst, Math.abs(n1 - n0) / n0);
    }
    worst <= tolerance
      ? pass("范数保持 ‖RoPE(x,m)‖ = ‖x‖", `最大相对偏差 ${exp(worst)}`)
      : fail("范数保持", `最大相对偏差 ${exp(worst)} 超过容差`);
  }

  /* 3. 零位置恒等 */
  {
    const x = randomVec(dim, rng);
    const y = applyRoPEToVector(x, 0, { base, layout, scaling });
    let worst = 0;
    for (let i = 0; i < dim; i++) worst = Math.max(worst, Math.abs(x[i] - y[i]));
    worst <= tolerance
      ? pass("零位置恒等 RoPE(x,0) = x", `最大偏差 ${exp(worst)}`)
      : fail("零位置恒等", `最大偏差 ${exp(worst)} 超过容差`);
  }

  /* 4. 相对位置不变性：整体平移位置，内积不变 */
  {
    const q = randomVec(dim, rng);
    const k = randomVec(dim, rng);
    const pairs = [
      [3, 11],
      [0, 64],
      [100, 101],
      [512, 900],
    ];
    let worst = 0;
    for (const [m, n] of pairs) {
      const ref = ropeInnerProduct(q, k, m, n, { base, layout, scaling });
      for (const t of [1, 17, 500]) {
        const shifted = ropeInnerProduct(q, k, m + t, n + t, { base, layout, scaling });
        worst = Math.max(worst, Math.abs(shifted - ref) / Math.max(1e-12, Math.abs(ref)));
      }
    }
    worst <= 1e-8
      ? pass("相对位置不变性", `平移后最大相对偏差 ${exp(worst)}`)
      : fail("相对位置不变性", `平移后最大相对偏差 ${exp(worst)}`);
  }

  /* 5. 内积只依赖位置差：⟨R_m q, R_n k⟩ = ⟨R_{m−n} q, k⟩ */
  {
    const q = randomVec(dim, rng);
    const k = randomVec(dim, rng);
    let worst = 0;
    for (const [m, n] of [
      [5, 2],
      [40, 33],
      [300, 0],
    ]) {
      const lhs = ropeInnerProduct(q, k, m, n, { base, layout, scaling });
      const rhs = ropeInnerProduct(q, k, m - n, 0, { base, layout, scaling });
      worst = Math.max(worst, Math.abs(lhs - rhs) / Math.max(1e-12, Math.abs(lhs)));
    }
    worst <= 1e-8
      ? pass("内积只依赖位置差", `最大相对偏差 ${exp(worst)}`)
      : fail("内积只依赖位置差", `最大相对偏差 ${exp(worst)}`);
  }

  /* 6. 可逆性：−m 旋转可还原 */
  {
    const x = randomVec(dim, rng);
    let worst = 0;
    for (const m of positions) {
      const y = applyRoPEToVector(x, m, { base, layout, scaling });
      const back = applyRoPEToVector(y, -m, { base, layout, scaling });
      for (let i = 0; i < dim; i++) worst = Math.max(worst, Math.abs(back[i] - x[i]));
    }
    worst <= 1e-9
      ? pass("可逆性 旋转 −m 可还原", `最大偏差 ${exp(worst)}`)
      : fail("可逆性", `最大偏差 ${exp(worst)}`);
  }

  return { passed: checks.every((c) => c.passed), checks };
}
