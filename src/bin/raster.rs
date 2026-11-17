//! 纯 CPU 单线程：光栅化"覆盖判定"的两种写法对比。
//!
//! - **逐像素 edge function**：传统软件光栅化，每个像素三次叉积。
//! - **tile 三态 + 位掩码**：先把屏幕切成 8×8 tile，用 4 个角把每个 tile 分成
//!   三态——全外（跳过）/ 全内（一次记 64 像素）/ 混合（退回逐像素）。
//!   凸性保证这个分类是**精确**的：三角形凸、tile 凸，角点全在内部则整个 tile
//!   在内部；某条边把 4 个角全留在外侧则整个 tile 在外侧。
//!
//! 两种实现的覆盖像素总数必须完全相等——这是正确性断言，不是估计。

use std::hint::black_box;
use std::time::Instant;

const W: usize = 1920;
const H: usize = 1080;
const TILE: usize = 8;
const TW: usize = W / TILE;
const TH: usize = H / TILE;
const TRIS: usize = 100_000;

#[derive(Clone, Copy)]
struct Tri {
    v: [[f32; 2]; 3],
}

#[inline(always)]
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }
}

/// 一半小三角形（粒子/UI），一半大三角形（地形/墙面）
fn gen_tris() -> Vec<Tri> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut tris = Vec::with_capacity(TRIS);
    for i in 0..TRIS {
        let cx = rng.f() * W as f32;
        let cy = rng.f() * H as f32;
        let r = if i % 2 == 0 {
            2.0 + rng.f() * 4.0 // 小：半径 2..6
        } else {
            18.0 + rng.f() * 62.0 // 大：半径 18..80
        };
        let a0 = rng.f() * std::f32::consts::TAU;
        let a1 = a0 + 2.0 * std::f32::consts::TAU / 3.0;
        let a2 = a1 + 2.0 * std::f32::consts::TAU / 3.0;
        let p = |a: f32| [cx + r * a.cos(), cy + r * a.sin()];
        let mut t = Tri {
            v: [p(a0), p(a1), p(a2)],
        };
        // 显式保证代数平面上的逆时针：内部点三条 edge 同为正。
        // 不依赖角度递增的隐含约定（屏幕坐标 y 向下时容易反）。
        let cross = (t.v[1][0] - t.v[0][0]) * (t.v[2][1] - t.v[0][1])
            - (t.v[1][1] - t.v[0][1]) * (t.v[2][0] - t.v[0][0]);
        if cross < 0.0 {
            t.v.swap(1, 2);
        }
        tris.push(t);
    }
    tris
}

/// 屏幕内的整数包围盒（像素中心判定范围）
fn bbox(t: &Tri) -> (usize, usize, usize, usize) {
    let mut x0 = f32::MAX;
    let mut y0 = f32::MAX;
    let mut x1 = f32::MIN;
    let mut y1 = f32::MIN;
    for v in &t.v {
        x0 = x0.min(v[0]);
        y0 = y0.min(v[1]);
        x1 = x1.max(v[0]);
        y1 = y1.max(v[1]);
    }
    let x0 = (x0.floor().max(0.0)) as usize;
    let y0 = (y0.floor().max(0.0)) as usize;
    let x1 = ((x1.ceil() as usize) + 1).min(W);
    let y1 = ((y1.ceil() as usize) + 1).min(H);
    (x0, y0, x1, y1)
}

fn tile_bbox(t: &Tri) -> (usize, usize, usize, usize) {
    let (x0, y0, x1, y1) = bbox(t);
    (
        (x0 / TILE).min(TW - 1),
        (y0 / TILE).min(TH - 1),
        ((x1 + TILE - 1) / TILE).min(TW),
        ((y1 + TILE - 1) / TILE).min(TH),
    )
}

/// 基线：逐像素三次 edge function
fn run_pixel(tris: &[Tri]) -> usize {
    let mut covered = 0usize;
    for tri in tris {
        let e = [
            (tri.v[0], tri.v[1]),
            (tri.v[1], tri.v[2]),
            (tri.v[2], tri.v[0]),
        ];
        let (x0, y0, x1, y1) = bbox(tri);
        for y in y0..y1 {
            let py = y as f32 + 0.5;
            for x in x0..x1 {
                let p = [x as f32 + 0.5, py];
                if edge(e[0].0, e[0].1, p) >= 0.0
                    && edge(e[1].0, e[1].1, p) >= 0.0
                    && edge(e[2].0, e[2].1, p) >= 0.0
                {
                    covered += 1;
                }
            }
        }
    }
    covered
}

/// tile 三态分类 + 位掩码计数
fn run_tile(tris: &[Tri]) -> (usize, [usize; 3]) {
    let mut covered = 0usize;
    let mut stats = [0usize; 3]; // [全外, 全内, 混合]
    for tri in tris {
        let e = [
            (tri.v[0], tri.v[1]),
            (tri.v[1], tri.v[2]),
            (tri.v[2], tri.v[0]),
        ];
        let (tx0, ty0, tx1, ty1) = tile_bbox(tri);
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let x0 = (tx * TILE) as f32;
                let y0 = (ty * TILE) as f32;
                let x1 = x0 + TILE as f32;
                let y1 = y0 + TILE as f32;
                let corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]];

                let mut outside = false;
                let mut all_inside = true;
                for (a, b) in e {
                    let mut any_inside = false;
                    let mut all_corner_inside = true;
                    for c in corners {
                        if edge(a, b, c) >= 0.0 {
                            any_inside = true;
                        } else {
                            all_corner_inside = false;
                        }
                    }
                    if !any_inside {
                        outside = true;
                        break;
                    }
                    if !all_corner_inside {
                        all_inside = false;
                    }
                }

                if outside {
                    stats[0] += 1;
                    continue;
                }
                if all_inside {
                    // 整块 64 像素一次计入（凸性保证精确）
                    stats[1] += 1;
                    covered += TILE * TILE;
                    continue;
                }
                // 混合 tile：只有这里退回逐像素
                stats[2] += 1;
                let px0 = tx * TILE;
                let py0 = ty * TILE;
                for y in py0..py0 + TILE {
                    let py = y as f32 + 0.5;
                    for x in px0..px0 + TILE {
                        let p = [x as f32 + 0.5, py];
                        if edge(e[0].0, e[0].1, p) >= 0.0
                            && edge(e[1].0, e[1].1, p) >= 0.0
                            && edge(e[2].0, e[2].1, p) >= 0.0
                        {
                            covered += 1;
                        }
                    }
                }
            }
        }
    }
    (covered, stats)
}

fn main() {
    let tris = gen_tris();
    let tris = black_box(tris);

    let mut best_pixel = f64::INFINITY;
    let mut covered_pixel = 0usize;
    for _ in 0..5 {
        let t = Instant::now();
        covered_pixel = black_box(run_pixel(&tris));
        best_pixel = best_pixel.min(t.elapsed().as_secs_f64());
    }

    let mut best_tile = f64::INFINITY;
    let mut covered_tile = 0usize;
    let mut stats = [0usize; 3];
    for _ in 0..5 {
        let t = Instant::now();
        let (c, s) = run_tile(&tris);
        best_tile = best_tile.min(t.elapsed().as_secs_f64());
        covered_tile = c;
        stats = s;
    }

    assert_eq!(
        covered_pixel, covered_tile,
        "两种实现的覆盖像素必须完全相等（凸性分类是精确的）"
    );

    println!("屏幕 {W}×{H}，tile {TILE}×{TILE}（{TW}×{TH} = {} 块）", TW * TH);
    println!("三角形 {TRIS}（小 50% / 大 50%），单线程 release\n");
    println!("覆盖像素合计 {covered_pixel}（两种实现一致 ✓）\n");

    let per = |secs: f64| secs * 1e9 / TRIS as f64;
    let mps = |secs: f64| TRIS as f64 / secs / 1e6;
    println!(
        "{:<28} | {:>10} | {:>12} | {:>14}",
        "实现", "总耗时", "ns/三角形", "百万三角形/秒"
    );
    println!("{:-<28}-+-{:-<10}-+-{:-<12}-+-{:-<14}", "", "", "", "");
    println!(
        "{:<28} | {:>8.2} ms | {:>12.1} | {:>14.2}",
        "逐像素 edge function",
        best_pixel * 1000.0,
        per(best_pixel),
        mps(best_pixel)
    );
    println!(
        "{:<28} | {:>8.2} ms | {:>12.1} | {:>14.2}",
        "tile 三态 + 位掩码",
        best_tile * 1000.0,
        per(best_tile),
        mps(best_tile)
    );
    println!(
        "\n倍数：{:.2}×（逐像素 / tile）",
        best_pixel / best_tile
    );
    println!(
        "tile 分类：全外 {}（{:.1}%）· 全内 {}（{:.1}%）· 混合 {}（{:.1}%）",
        stats[0],
        stats[0] as f64 / (stats[0] + stats[1] + stats[2]) as f64 * 100.0,
        stats[1],
        stats[1] as f64 / (stats[0] + stats[1] + stats[2]) as f64 * 100.0,
        stats[2],
        stats[2] as f64 / (stats[0] + stats[1] + stats[2]) as f64 * 100.0
    );
    println!(
        "其中「全内」tile 一次结算 {} 像素，占总覆盖的 {:.1}%",
        TILE * TILE,
        stats[1] as f64 * (TILE * TILE) as f64 / covered_pixel as f64 * 100.0
    );
}
