import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  applyRoPE,
  applyRoPEToVector,
  ropeFrequencies,
  ropeMatrix,
  ropeInnerProduct,
  verifyRoPE,
  LAYOUTS,
  DEFAULT_BASE,
} from "../src/index.js";

/* ------------------------------------------------------------------ 工具 */

const TOL = 1e-12;

function norm(v) {
  let acc = 0;
  for (let i = 0; i < v.length; i++) acc += v[i] * v[i];
  return Math.sqrt(acc);
}

function dot(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc += a[i] * b[i];
  return acc;
}

function approx(actual, expected, tol = TOL, msg) {
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `${msg ?? "数值不符"}：期望 ${expected}，实得 ${actual}（差 ${Math.abs(actual - expected)}）`
  );
}

/** 矩阵（行主序 dim×dim）乘向量 */
function matVec(M, v, dim) {
  const out = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    let acc = 0;
    for (let j = 0; j < dim; j++) acc += M[i * dim + j] * v[j];
    out[i] = acc;
  }
  return out;
}

/* ------------------------------------------------------------------ 频率表 */

describe("ropeFrequencies", () => {
  it("长度为 dim/2", () => {
    assert.equal(ropeFrequencies(2).length, 1);
    assert.equal(ropeFrequencies(64).length, 32);
    assert.equal(ropeFrequencies(128).length, 64);
  });

  it("第 0 项恒为 1（base^0）", () => {
    approx(ropeFrequencies(64)[0], 1, TOL, "θ₀");
    approx(ropeFrequencies(8, { base: 500 })[0], 1, TOL, "θ₀ 与 base 无关");
  });

  it("θ_i = base^(−2i/d)", () => {
    const dim = 16;
    const base = 10000;
    const f = ropeFrequencies(dim, { base });
    for (let i = 0; i < dim / 2; i++) {
      approx(f[i], Math.pow(base, (-2 * i) / dim), TOL, `θ_${i}`);
    }
  });

  it("严格递减（波长递增）", () => {
    const f = ropeFrequencies(64);
    for (let i = 1; i < f.length; i++) {
      assert.ok(f[i] < f[i - 1], `θ_${i} 应小于 θ_${i - 1}`);
    }
  });

  it("dim 必须是正偶数", () => {
    assert.throws(() => ropeFrequencies(0), /正整数/);
    assert.throws(() => ropeFrequencies(-4), /正整数/);
    assert.throws(() => ropeFrequencies(7), /偶数/);
    assert.throws(() => ropeFrequencies(3.5), /正整数/);
  });

  it("ntk 缩放让频率变小（波长变长）", () => {
    const dim = 64;
    const plain = ropeFrequencies(dim);
    const ntk = ropeFrequencies(dim, { scaling: { type: "ntk", factor: 4 } });
    for (let i = 1; i < plain.length; i++) {
      assert.ok(ntk[i] < plain[i], `ntk 下 θ_${i} 应变小`);
    }
  });

  it("ntk 的 factor 必须大于 1", () => {
    assert.throws(() => ropeFrequencies(64, { scaling: { type: "ntk", factor: 1 } }), /大于 1/);
    assert.throws(() => ropeFrequencies(64, { scaling: { type: "ntk", factor: 0.5 } }), /大于 1/);
  });

  it("未知的 scaling.type 报错", () => {
    assert.throws(() => ropeFrequencies(64, { scaling: { type: "yarn" } }), /未知的 scaling/);
  });
});

/* ------------------------------------------------------------------ 旋转矩阵 */

describe("ropeMatrix", () => {
  it("是正交矩阵：RᵀR = I", () => {
    const dim = 16;
    for (const layout of LAYOUTS) {
      for (const m of [0, 1, 13, 512]) {
        const R = ropeMatrix(m, dim, { layout });
        for (let i = 0; i < dim; i++) {
          for (let j = 0; j < dim; j++) {
            let acc = 0;
            for (let k = 0; k < dim; k++) acc += R[k * dim + i] * R[k * dim + j];
            approx(acc, i === j ? 1 : 0, 1e-12, `RᵀR[${i},${j}] @ m=${m} layout=${layout}`);
          }
        }
      }
    }
  });

  it("位置 0 时是单位矩阵", () => {
    const dim = 8;
    const R = ropeMatrix(0, dim);
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        approx(R[i * dim + j], i === j ? 1 : 0, TOL, `I[${i},${j}]`);
      }
    }
  });

  it("矩阵乘向量与 applyRoPE 结果一致（交叉验证）", () => {
    const dim = 12;
    const v = Float64Array.from({ length: dim }, (_, i) => Math.sin(i * 1.7) * 2);
    for (const layout of LAYOUTS) {
      for (const m of [0, 3, 47, 999]) {
        const viaMatrix = matVec(ropeMatrix(m, dim, { layout }), v, dim);
        const viaApi = applyRoPEToVector(v, m, { layout });
        for (let i = 0; i < dim; i++) {
          approx(viaApi[i], viaMatrix[i], 1e-12, `dim=${dim} layout=${layout} m=${m} 第 ${i} 维`);
        }
      }
    }
  });

  it("非法 layout 报错", () => {
    assert.throws(() => ropeMatrix(1, 8, { layout: "neox" }), /layout/);
  });
});

/* ------------------------------------------------------------------ 手动对照 */

describe("applyRoPE 的闭式对照", () => {
  it("dim=2 时就是标准二维旋转", () => {
    // dim=2 → half=1 → θ₀ = base^0 = 1，所以旋转角恰好等于位置 m
    const x = Float64Array.from([1, 0]);
    const m = 0.5; // 用小数位置方便对照 cos/sin
    const y = applyRoPEToVector(x, m);
    approx(y[0], Math.cos(m), TOL, "x'₀");
    approx(y[1], Math.sin(m), TOL, "x'₁");
  });

  it("dim=4 的 half 布局：配对是 (0,2) 与 (1,3)", () => {
    const dim = 4;
    const x = Float64Array.from([1, 0, 0, 0]);
    const m = 1;
    const f = ropeFrequencies(dim);
    const y = applyRoPEToVector(x, m, { layout: "half" });
    // 第 0 组 (x₀, x₂) 参与旋转，第 1 组 (x₁, x₃) 全零不受影响
    approx(y[0], Math.cos(m * f[0]), TOL, "y₀");
    approx(y[2], Math.sin(m * f[0]), TOL, "y₂");
    approx(y[1], 0, TOL, "y₁ 应保持 0");
    approx(y[3], 0, TOL, "y₃ 应保持 0");
  });

  it("dim=4 的 interleaved 布局：配对是 (0,1) 与 (2,3)", () => {
    const dim = 4;
    const x = Float64Array.from([1, 0, 0, 0]);
    const m = 1;
    const f = ropeFrequencies(dim);
    const y = applyRoPEToVector(x, m, { layout: "interleaved" });
    approx(y[0], Math.cos(m * f[0]), TOL, "y₀");
    approx(y[1], Math.sin(m * f[0]), TOL, "y₁");
    approx(y[2], 0, TOL, "y₂ 应保持 0");
    approx(y[3], 0, TOL, "y₃ 应保持 0");
  });

  it("dim=2 时两种布局等价", () => {
    const x = Float64Array.from([0.3, -0.7]);
    const a = applyRoPEToVector(x, 5, { layout: "half" });
    const b = applyRoPEToVector(x, 5, { layout: "interleaved" });
    for (let i = 0; i < 2; i++) approx(a[i], b[i], TOL, `第 ${i} 维`);
  });

  it("dim>2 时两种布局结果不同", () => {
    const dim = 8;
    const x = Float64Array.from({ length: dim }, (_, i) => i + 1);
    const a = applyRoPEToVector(x, 3, { layout: "half" });
    const b = applyRoPEToVector(x, 3, { layout: "interleaved" });
    let same = true;
    for (let i = 0; i < dim; i++) if (Math.abs(a[i] - b[i]) > 1e-9) same = false;
    assert.equal(same, false, "两种布局在 dim>2 时应当不同");
  });
});

/* ------------------------------------------------------------------ 核心性质 */

describe("核心性质", () => {
  for (const layout of LAYOUTS) {
    it(`[${layout}] 位置 0 是恒等变换`, () => {
      const dim = 32;
      const x = Float64Array.from({ length: dim }, (_, i) => Math.cos(i) * 3);
      const y = applyRoPEToVector(x, 0, { layout });
      for (let i = 0; i < dim; i++) approx(y[i], x[i], TOL, `第 ${i} 维`);
    });

    it(`[${layout}] 保持范数`, () => {
      const dim = 32;
      const x = Float64Array.from({ length: dim }, (_, i) => Math.sin(i * 2.3));
      const n0 = norm(x);
      for (const m of [1, 10, 100, 5000]) {
        approx(norm(applyRoPEToVector(x, m, { layout })), n0, 1e-12, `m=${m}`);
      }
    });

    it(`[${layout}] 内积只依赖相对位置`, () => {
      const dim = 32;
      const q = Float64Array.from({ length: dim }, (_, i) => Math.sin(i));
      const k = Float64Array.from({ length: dim }, (_, i) => Math.cos(i * 1.3));
      const ref = ropeInnerProduct(q, k, 7, 3, { layout });
      for (const t of [1, 64, 1000]) {
        approx(
          ropeInnerProduct(q, k, 7 + t, 3 + t, { layout }),
          ref,
          1e-9,
          `平移 ${t} 后内积应不变`
        );
      }
      // 位置差形式：⟨R_m q, R_n k⟩ = ⟨R_{m−n} q, k⟩
      approx(ropeInnerProduct(q, k, 7, 3, { layout }), ropeInnerProduct(q, k, 4, 0, { layout }), 1e-9);
    });

    it(`[${layout}] 旋转可逆`, () => {
      const dim = 16;
      const x = Float64Array.from({ length: dim }, (_, i) => i * 0.37 - 1);
      const y = applyRoPEToVector(x, 123, { layout });
      const back = applyRoPEToVector(y, -123, { layout });
      for (let i = 0; i < dim; i++) approx(back[i], x[i], 1e-12, `第 ${i} 维`);
    });
  }
});

/* ------------------------------------------------------------------ 形状处理 */

describe("形状处理", () => {
  const dim = 8;

  it("[seq, dim]：每一行按自己的位置旋转", () => {
    const seq = 4;
    const data = Float64Array.from({ length: seq * dim }, (_, i) => (i % dim) + 1);
    const out = applyRoPE(data, [seq, dim]);
    for (let s = 0; s < seq; s++) {
      const row = data.slice(s * dim, (s + 1) * dim);
      const expected = applyRoPEToVector(row, s);
      for (let i = 0; i < dim; i++) {
        approx(out[s * dim + i], expected[i], 1e-12, `位置 ${s} 第 ${i} 维`);
      }
    }
  });

  it("[heads, seq, dim]：所有 head 共用同一套位置", () => {
    const heads = 3;
    const seq = 5;
    const data = Float64Array.from({ length: heads * seq * dim }, (_, i) => Math.sin(i));
    const out = applyRoPE(data, [heads, seq, dim]);
    for (let h = 0; h < heads; h++) {
      for (let s = 0; s < seq; s++) {
        const base = (h * seq + s) * dim;
        const row = data.slice(base, base + dim);
        const expected = applyRoPEToVector(row, s);
        for (let i = 0; i < dim; i++) {
          approx(out[base + i], expected[i], 1e-12, `head ${h} 位置 ${s} 第 ${i} 维`);
        }
      }
    }
  });

  it("[batch, heads, seq, dim]：四维也正确", () => {
    const shape = [2, 3, 4, dim];
    const total = shape.reduce((a, b) => a * b, 1);
    const data = Float64Array.from({ length: total }, (_, i) => Math.cos(i * 0.1));
    const out = applyRoPE(data, shape);
    assert.equal(out.length, total);
    // 抽查 batch=1, head=2, seq=3
    const idx = ((1 * 3 + 2) * 4 + 3) * dim;
    const expected = applyRoPEToVector(data.slice(idx, idx + dim), 3);
    for (let i = 0; i < dim; i++) approx(out[idx + i], expected[i], 1e-12, `抽查第 ${i} 维`);
  });

  it("不修改入参", () => {
    const data = Float64Array.from({ length: 4 * dim }, (_, i) => i + 1);
    const copy = Float64Array.from(data);
    applyRoPE(data, [4, dim]);
    assert.deepEqual(Array.from(data), Array.from(copy), "入参不应被修改");
  });

  it("data 长度与 shape 不符时报错", () => {
    assert.throws(() => applyRoPE(new Float64Array(10), [2, 8]), /不符/);
  });

  it("shape 非法时报错", () => {
    assert.throws(() => applyRoPE(new Float64Array(8), []), /非空数组/);
    assert.throws(() => applyRoPE(new Float64Array(8), [0, 8]), /正整数/);
    assert.throws(() => applyRoPE(new Float64Array(8), [2, -4]), /正整数/);
  });

  it("dim 为奇数时报错", () => {
    assert.throws(() => applyRoPE(new Float64Array(6), [2, 3]), /偶数/);
  });
});

/* ------------------------------------------------------------------ 位置与偏移 */

describe("位置与偏移", () => {
  const dim = 8;
  const x = Float64Array.from({ length: dim }, (_, i) => i * 0.5 + 1);

  it("offset 把位置整体后移", () => {
    const a = applyRoPE(x, [1, dim], { offset: 10 });
    const b = applyRoPEToVector(x, 10);
    for (let i = 0; i < dim; i++) approx(a[i], b[i], TOL, `第 ${i} 维`);
  });

  it("positions 可显式指定每个位置", () => {
    const data = Float64Array.from({ length: 3 * dim }, (_, i) => (i % dim) + 1);
    const out = applyRoPE(data, [3, dim], { positions: [100, 5, 42] });
    for (const [s, pos] of [100, 5, 42].entries()) {
      const row = data.slice(s * dim, (s + 1) * dim);
      const expected = applyRoPEToVector(row, pos);
      for (let i = 0; i < dim; i++) {
        approx(out[s * dim + i], expected[i], 1e-12, `第 ${s} 行（位置 ${pos}）第 ${i} 维`);
      }
    }
  });

  it("positions 长度不符时报错", () => {
    const data = new Float64Array(3 * dim);
    assert.throws(() => applyRoPE(data, [3, dim], { positions: [1, 2] }), /长度必须是 3/);
  });

  it("offset 非整数时报错", () => {
    assert.throws(() => applyRoPE(x, [1, dim], { offset: 1.5 }), /整数/);
  });

  it("增量解码：offset 拼接等价于一次性长序列", () => {
    // 这是 KV cache 场景的实际用法——先算前 3 个 token，再算第 4 个
    const seq = 4;
    const data = Float64Array.from({ length: seq * dim }, (_, i) => Math.sin(i * 0.7));
    const whole = applyRoPE(data, [seq, dim]);

    const head = applyRoPE(data.slice(0, 3 * dim), [3, dim], { offset: 0 });
    const tail = applyRoPE(data.slice(3 * dim), [1, dim], { offset: 3 });

    for (let i = 0; i < 3 * dim; i++) approx(head[i], whole[i], TOL, `前缀第 ${i} 维`);
    for (let i = 0; i < dim; i++) approx(tail[i], whole[3 * dim + i], TOL, `增量第 ${i} 维`);
  });
});

/* ------------------------------------------------------------------ 上下文扩展 */

describe("上下文扩展", () => {
  const dim = 16;
  const x = Float64Array.from({ length: dim }, (_, i) => Math.sin(i));

  it("linear 缩放等价于把位置除以 factor", () => {
    const a = applyRoPEToVector(x, 64, { scaling: { type: "linear", factor: 4 } });
    const b = applyRoPEToVector(x, 16);
    for (let i = 0; i < dim; i++) approx(a[i], b[i], 1e-12, `第 ${i} 维`);
  });

  it("linear 缩放不改变 base", () => {
    const plain = ropeFrequencies(dim);
    const scaled = ropeFrequencies(dim, { scaling: { type: "linear", factor: 8 } });
    for (let i = 0; i < plain.length; i++) approx(scaled[i], plain[i], TOL, `θ_${i}`);
  });

  it("缩放后仍保持范数与相对位置性质", () => {
    const scaling = { type: "linear", factor: 4 };
    const n0 = norm(x);
    for (const m of [0, 32, 128]) {
      approx(norm(applyRoPEToVector(x, m, { scaling })), n0, 1e-12, `m=${m} 范数`);
    }
    const q = x;
    const k = Float64Array.from({ length: dim }, (_, i) => Math.cos(i));
    const ref = ropeInnerProduct(q, k, 20, 5, { scaling });
    approx(ropeInnerProduct(q, k, 40, 25, { scaling }), ref, 1e-9, "平移后内积应不变");
  });

  it("ntk 与 linear 得到不同结果", () => {
    const a = applyRoPEToVector(x, 64, { scaling: { type: "ntk", factor: 4 } });
    const b = applyRoPEToVector(x, 64, { scaling: { type: "linear", factor: 4 } });
    let same = true;
    for (let i = 0; i < dim; i++) if (Math.abs(a[i] - b[i]) > 1e-9) same = false;
    assert.equal(same, false, "两种缩放策略应给出不同结果");
  });
});

/* ------------------------------------------------------------------ 内置自检 */

describe("verifyRoPE 内置自检", () => {
  it("默认配置全部通过", () => {
    const r = verifyRoPE();
    assert.equal(r.passed, true, JSON.stringify(r.checks.filter((c) => !c.passed), null, 2));
    assert.equal(r.checks.length, 6);
  });

  it("两种 layout 都通过", () => {
    for (const layout of LAYOUTS) {
      const r = verifyRoPE({ layout });
      assert.equal(r.passed, true, `layout=${layout} 应通过`);
    }
  });

  it("多种 dim 都通过", () => {
    for (const dim of [2, 4, 16, 64, 128]) {
      const r = verifyRoPE({ dim });
      assert.equal(r.passed, true, `dim=${dim} 应通过`);
    }
  });

  it("带上下文扩展也通过", () => {
    for (const scaling of [
      { type: "linear", factor: 4 },
      { type: "ntk", factor: 4 },
    ]) {
      const r = verifyRoPE({ scaling });
      assert.equal(r.passed, true, `scaling=${scaling.type} 应通过`);
    }
  });

  it("每项检查都带可读的 detail", () => {
    const r = verifyRoPE();
    for (const c of r.checks) {
      assert.equal(typeof c.name, "string");
      assert.equal(typeof c.detail, "string");
      assert.ok(c.detail.length > 0, `${c.name} 的 detail 不应为空`);
    }
  });
});

/* -------------------------------------------------- 接错布局会怎样 */

/**
 * 这两种对接错误在真实集成里都会出现，而且**不报错**：形状对得上、
 * 数值有界、注意力照常算，只是效果静默变差。用测试把它们钉住，
 * 以后改动不能悄悄让「错配 ≈ 正确」，否则一旦发生就再也查不出来。
 */
describe("接错配对约定会静默出错", () => {
  it("half 布局的数据用 interleaved 去旋转，结果明显不同", () => {
    const dim = 8;
    const x = Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    const right = applyRoPEToVector(x, 5, { layout: "half" });
    const wrong = applyRoPEToVector(x, 5, { layout: "interleaved" });
    let worst = 0;
    for (let i = 0; i < dim; i++) worst = Math.max(worst, Math.abs(right[i] - wrong[i]));
    assert.ok(worst > 0.5, `错配后应产生明显差异，实得最大差 ${worst}`);
  });

  it("错配后范数仍然守恒 —— 所以光看范数根本发现不了", () => {
    const x = Float64Array.from([0.3, -1.4, 2.2, 0.7, -0.9, 1.1, 3.0, -0.2]);
    const n = Math.hypot(...x);
    for (const layout of ["half", "interleaved"]) {
      const y = applyRoPEToVector(x, 37, { layout });
      assert.ok(Math.abs(Math.hypot(...y) - n) < 1e-12, `${layout} 下范数应守恒`);
    }
  });

  it("错配后矩阵依然正交 —— 每条数学性质都过，只有语义是错的", () => {
    const dim = 8;
    const R = ropeMatrix(13, dim, { layout: "interleaved" });
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        let acc = 0;
        for (let k = 0; k < dim; k++) acc += R[k * dim + i] * R[k * dim + j];
        assert.ok(Math.abs(acc - (i === j ? 1 : 0)) < 1e-12);
      }
    }
  });
});

/* ------------------------------------------------------------------ 常量 */

describe("导出常量", () => {
  it("LAYOUTS 包含两种约定", () => {
    assert.deepEqual([...LAYOUTS], ["half", "interleaved"]);
  });

  it("DEFAULT_BASE 是 10000", () => {
    assert.equal(DEFAULT_BASE, 10000);
  });
});
