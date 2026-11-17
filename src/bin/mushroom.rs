//! 桌面程序：**3D 三角形** + 可转镜头，两套 CPU 光栅化实时对比。
//!
//! - 默认 `--plain`：逐像素三次 edge function + 逐像素深度测试
//! - `--ternary`：tile 三态分类（全外跳过 / 全内跳过 edge） + 位掩码
//!
//! 场景：球壳内随机分布的 3D 三角形（每片有独立朝向与颜色），透视投影 + 深度缓冲。
//! 交互：**按住左键拖动旋转镜头**，滚轮靠近/远离，`1..4` 切三角形数，`Esc` 退出。
//!
//! 画面左上角与窗口标题实时显示：FPS / 帧时间 / 光栅化耗时 / 三角形数 / 可见片数 / 镜头角。

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const W: usize = 960;
const H: usize = 540;
const TILE: usize = 8;
const TW: usize = W / TILE;
const TH: usize = H / TILE;
const NEAR: f32 = 0.1;
const FAR: f32 = 100.0;
const TIERS: [usize; 4] = [10_000, 30_000, 60_000, 120_000];
const OVERLAY_W: u32 = 1024;
const OVERLAY_H: u32 = 48;

#[derive(Clone, Copy)]
struct Tri3 {
    v: [[f32; 3]; 3],
    color: [u8; 4],
}

/// 投影后的屏幕三角形
#[derive(Clone, Copy)]
struct ScreenTri {
    p: [[f32; 2]; 3],
    d: [f32; 3],
    color: [u8; 4],
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

/// 蘑菇：伞盖（压扁半球）+ 菌柄（圆柱）+ 伞底圆盘，总计约 `total` 片三角形。
/// 这就是「显卡蘑菇测试」的几何体——三万片小三角形拼出的一只蘑菇。
fn gen_mushroom(total: usize) -> Vec<Tri3> {
    let mut tris = Vec::with_capacity(total + 64);

    // 毒蝇伞的白斑：在 (u, v) 参数空间里预生成若干圆斑
    let mut rng = Rng(0xDEAD_BEEF_1234_5678);
    let mut spots: Vec<(f32, f32, f32)> = Vec::new();
    for _ in 0..46 {
        spots.push((
            rng.f(),
            0.05 + rng.f() * 0.78,
            0.016 + rng.f() * 0.030,
        ));
    }
    let cap_base_y = 0.35f32; // 伞盖底部高度
    let cap_h = 1.55f32; // 伞盖高度
    let cap_r = 2.45f32; // 伞盖半径
    let stem_r = 0.48f32; // 菌柄半径
    let stem_top = cap_base_y;
    let stem_bot = -2.0f32;

    let cap_tris = total * 70 / 100;
    let stem_tris = total * 20 / 100;
    let disc_tris = total.saturating_sub(cap_tris + stem_tris);

    // ---- 伞盖：压扁半球 ----
    let cells = (cap_tris / 2).max(1);
    let nu = ((cells as f32 * 2.0).sqrt() as usize).max(8);
    let nv = (cells / nu).max(4);
    let cap = |u: f32, v: f32| -> [f32; 3] {
        let phi = u * std::f32::consts::TAU;
        let theta = v * std::f32::consts::FRAC_PI_2;
        let r = cap_r * theta.cos();
        [
            r * phi.cos(),
            cap_base_y + cap_h * theta.sin(),
            r * phi.sin(),
        ]
    };
    for j in 0..nv {
        for i in 0..nu {
            let u0 = i as f32 / nu as f32;
            let u1 = (i + 1) as f32 / nu as f32;
            let v0 = j as f32 / nv as f32;
            let v1 = (j + 1) as f32 / nv as f32;
            let p00 = cap(u0, v0);
            let p10 = cap(u1, v0);
            let p01 = cap(u0, v1);
            let p11 = cap(u1, v1);
            // 毒蝇伞：深红伞盖 + 白斑点
            let uc = (u0 + u1) * 0.5;
            let vc = (v0 + v1) * 0.5;
            let mut in_spot = false;
            for (su, sv, sr) in &spots {
                let mut du = (uc - su).abs();
                if du > 0.5 {
                    du = 1.0 - du; // 经度环绕
                }
                let dv = vc - sv;
                if (du * du + dv * dv).sqrt() < *sr {
                    in_spot = true;
                    break;
                }
            }
            let color = if in_spot {
                [248, 246, 238, 255]
            } else {
                // 中心更红，边缘略暗，做出弧度
                let t = 0.55 + 0.45 * (1.0 - v0);
                [
                    (182.0 + 60.0 * t) as u8,
                    (30.0 + 20.0 * (1.0 - t)) as u8,
                    (28.0 + 18.0 * (1.0 - t)) as u8,
                    255,
                ]
            };
            tris.push(Tri3 {
                v: [p00, p10, p11],
                color,
            });
            tris.push(Tri3 {
                v: [p00, p11, p01],
                color,
            });
        }
    }

    // ---- 菌柄：圆柱侧面 ----
    let stem_nu = 96usize;
    let stem_nv = (stem_tris / 2 / stem_nu).max(4);
    let stem = |u: f32, v: f32| -> [f32; 3] {
        let phi = u * std::f32::consts::TAU;
        let y = stem_bot + (stem_top - stem_bot) * v;
        [stem_r * phi.cos(), y, stem_r * phi.sin()]
    };
    for j in 0..stem_nv {
        for i in 0..stem_nu {
            let u0 = i as f32 / stem_nu as f32;
            let u1 = (i + 1) as f32 / stem_nu as f32;
            let v0 = j as f32 / stem_nv as f32;
            let v1 = (j + 1) as f32 / stem_nv as f32;
            let p00 = stem(u0, v0);
            let p10 = stem(u1, v0);
            let p01 = stem(u0, v1);
            let p11 = stem(u1, v1);
            let s = (175.0 + 65.0 * v0) as u8;
            let color = [
                s,
                (s as f32 * 0.95) as u8,
                (s as f32 * 0.84) as u8,
                255,
            ];
            tris.push(Tri3 {
                v: [p00, p10, p11],
                color,
            });
            tris.push(Tri3 {
                v: [p00, p11, p01],
                color,
            });
        }
    }

    // ---- 伞底圆盘（环形，从菌柄到伞缘）----
    let disc_nu = 96usize;
    let disc_nv = (disc_tris / 2 / disc_nu).max(2);
    for j in 0..disc_nv {
        for i in 0..disc_nu {
            let u0 = i as f32 / disc_nu as f32;
            let u1 = (i + 1) as f32 / disc_nu as f32;
            let r0 = stem_r + (cap_r - stem_r) * (j as f32 / disc_nv as f32);
            let r1 = stem_r + (cap_r - stem_r) * ((j + 1) as f32 / disc_nv as f32);
            let phi0 = u0 * std::f32::consts::TAU;
            let phi1 = u1 * std::f32::consts::TAU;
            let p = |r: f32, a: f32| [r * a.cos(), cap_base_y, r * a.sin()];
            let p00 = p(r0, phi0);
            let p10 = p(r0, phi1);
            let p01 = p(r1, phi0);
            let p11 = p(r1, phi1);
            let color = [88u8, 32, 30, 255];
            tris.push(Tri3 {
                v: [p00, p10, p11],
                color,
            });
            tris.push(Tri3 {
                v: [p00, p11, p01],
                color,
            });
        }
    }

    tris
}

// ---------- 矩阵 ----------

type M4 = [[f32; 4]; 4];

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> M4 {
    let f = 1.0 / (fov_y / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> M4 {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f32; 3]| {
        let l = dot(a, a).sqrt().max(1e-9);
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let f = norm(sub(center, eye));
    let s = norm(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn mul(a: M4, b: M4) -> M4 {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k][j] * b[i][k];
            }
            out[i][j] = sum;
        }
    }
    out
}

fn transform(m: M4, v: [f32; 3]) -> [f32; 4] {
    let mut o = [0.0f32; 4];
    for i in 0..4 {
        o[i] = m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2] + m[3][i];
    }
    o
}

/// 投影到屏幕坐标 + ndc 深度；返回 None 表示整片被丢弃（任意顶点在近平面后）
fn project(tris: &[Tri3], vp: M4) -> Vec<ScreenTri> {
    let mut out = Vec::with_capacity(tris.len());
    for t in tris {
        let mut clip = [[0.0f32; 4]; 3];
        let mut ok = true;
        for (i, v) in t.v.iter().enumerate() {
            let c = transform(vp, *v);
            if c[3] <= 1e-4 {
                ok = false;
                break;
            }
            clip[i] = c;
        }
        if !ok {
            continue;
        }
        let mut p = [[0.0f32; 2]; 3];
        let mut d = [0.0f32; 3];
        for i in 0..3 {
            let inv_w = 1.0 / clip[i][3];
            let nx = clip[i][0] * inv_w;
            let ny = clip[i][1] * inv_w;
            let nz = clip[i][2] * inv_w;
            p[i] = [
                (nx * 0.5 + 0.5) * W as f32,
                (0.5 - ny * 0.5) * H as f32,
            ];
            d[i] = nz;
        }
        // 屏幕空间绕序归一：背面（area < 0）交换两个顶点，让 edge 测试统一为 >= 0
        let area = edge(p[0], p[1], p[2]);
        if area == 0.0 {
            continue;
        }
        if area < 0.0 {
            p.swap(1, 2);
            d.swap(1, 2);
        }
        // 完全在屏幕外就跳过
        if p.iter().all(|q| q[0] < 0.0)
            || p.iter().all(|q| q[0] > W as f32)
            || p.iter().all(|q| q[1] < 0.0)
            || p.iter().all(|q| q[1] > H as f32)
        {
            continue;
        }
        out.push(ScreenTri {
            p,
            d,
            color: t.color,
        });
    }
    out
}

#[inline(always)]
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn bbox(t: &ScreenTri) -> (usize, usize, usize, usize) {
    let mut x0 = f32::MAX;
    let mut y0 = f32::MAX;
    let mut x1 = f32::MIN;
    let mut y1 = f32::MIN;
    for v in &t.p {
        x0 = x0.min(v[0]);
        y0 = y0.min(v[1]);
        x1 = x1.max(v[0]);
        y1 = y1.max(v[1]);
    }
    (
        (x0.floor().max(0.0)) as usize,
        (y0.floor().max(0.0)) as usize,
        ((x1.ceil() as usize) + 1).min(W),
        ((y1.ceil() as usize) + 1).min(H),
    )
}

#[inline(always)]
fn shade(color: [u8; 4], depth: f32) -> [u8; 4] {
    // 深度做雾化，让远近有层次
    let f = (1.0 - depth).clamp(0.15, 1.0);
    [
        (color[0] as f32 * f) as u8,
        (color[1] as f32 * f) as u8,
        (color[2] as f32 * f) as u8,
        255,
    ]
}

/// 纯 Rust：逐像素 edge + 逐像素深度
fn raster_plain(pixels: &mut [u8], zbuf: &mut [f32], tris: &[ScreenTri]) -> usize {
    let mut covered = 0usize;
    for tri in tris {
        let e = [(tri.p[0], tri.p[1]), (tri.p[1], tri.p[2]), (tri.p[2], tri.p[0])];
        let area = edge(tri.p[0], tri.p[1], tri.p[2]);
        if area == 0.0 {
            continue;
        }
        let inv_area = 1.0 / area;
        let (x0, y0, x1, y1) = bbox(tri);
        for y in y0..y1 {
            let py = y as f32 + 0.5;
            for x in x0..x1 {
                let p = [x as f32 + 0.5, py];
                let w0 = edge(e[1].0, e[1].1, p);
                let w1 = edge(e[2].0, e[2].1, p);
                let w2 = edge(e[0].0, e[0].1, p);
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                // 归一化重心，插值 ndc 深度
                let b0 = w0 * inv_area;
                let b1 = w1 * inv_area;
                let b2 = w2 * inv_area;
                let depth = b0 * tri.d[1] + b1 * tri.d[2] + b2 * tri.d[0];
                let i = y * W + x;
                if depth < zbuf[i] {
                    zbuf[i] = depth;
                    let c = shade(tri.color, depth);
                    pixels[i * 4..i * 4 + 4].copy_from_slice(&c);
                    covered += 1;
                }
            }
        }
    }
    covered
}

/// 三进制：tile 三态（全外跳过 / 全内跳过 edge）+ 位掩码
fn raster_ternary(pixels: &mut [u8], zbuf: &mut [f32], tris: &[ScreenTri]) -> usize {
    let mut covered = 0usize;
    for tri in tris {
        let e = [(tri.p[0], tri.p[1]), (tri.p[1], tri.p[2]), (tri.p[2], tri.p[0])];
        let area = edge(tri.p[0], tri.p[1], tri.p[2]);
        if area == 0.0 {
            continue;
        }
        let inv_area = 1.0 / area;
        let (x0, y0, x1, y1) = bbox(tri);
        let tx0 = (x0 / TILE).min(TW - 1);
        let ty0 = (y0 / TILE).min(TH - 1);
        let tx1 = ((x1 + TILE - 1) / TILE).min(TW);
        let ty1 = ((y1 + TILE - 1) / TILE).min(TH);
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let ax = (tx * TILE) as f32;
                let ay = (ty * TILE) as f32;
                let bx = ax + TILE as f32;
                let by = ay + TILE as f32;
                let corners = [[ax, ay], [bx, ay], [ax, by], [bx, by]];

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
                    continue;
                }
                if all_inside {
                    // 整块：跳过 edge，仍逐像素插值深度（3D 的深度不能省）
                    for y in ty * TILE..(ty + 1) * TILE {
                        let py = y as f32 + 0.5;
                        for x in tx * TILE..(tx + 1) * TILE {
                            let p = [x as f32 + 0.5, py];
                            let b0 = edge(e[1].0, e[1].1, p) * inv_area;
                            let b1 = edge(e[2].0, e[2].1, p) * inv_area;
                            let b2 = 1.0 - b0 - b1;
                            let depth = b0 * tri.d[1] + b1 * tri.d[2] + b2 * tri.d[0];
                            let i = y * W + x;
                            if depth < zbuf[i] {
                                zbuf[i] = depth;
                                let c = shade(tri.color, depth);
                                pixels[i * 4..i * 4 + 4].copy_from_slice(&c);
                                covered += 1;
                            }
                        }
                    }
                    continue;
                }
                // 混合 tile：逐像素 edge + 深度
                for y in ty * TILE..(ty + 1) * TILE {
                    let py = y as f32 + 0.5;
                    for x in tx * TILE..(tx + 1) * TILE {
                        let p = [x as f32 + 0.5, py];
                        let w0 = edge(e[1].0, e[1].1, p);
                        let w1 = edge(e[2].0, e[2].1, p);
                        let w2 = edge(e[0].0, e[0].1, p);
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        let b0 = w0 * inv_area;
                        let b1 = w1 * inv_area;
                        let b2 = w2 * inv_area;
                        let depth = b0 * tri.d[1] + b1 * tri.d[2] + b2 * tri.d[0];
                        let i = y * W + x;
                        if depth < zbuf[i] {
                            zbuf[i] = depth;
                            let c = shade(tri.color, depth);
                            pixels[i * 4..i * 4 + 4].copy_from_slice(&c);
                            covered += 1;
                        }
                    }
                }
            }
        }
    }
    covered
}

// ---------- shader（显示用） ----------

const TEX_WGSL: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let c = corners[vi];
    var o: VsOut;
    o.clip = vec4<f32>(c, 0.0, 1.0);
    o.uv = c * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return o;
}

@fragment
fn fs(v: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, v.uv);
}
"#;

const OVERLAY_WGSL: &str = r#"
struct Globals {
    size: vec2<f32>,
    _a: f32,
    _b: f32,
};
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_overlay(@builtin(vertex_index) vi: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    let px = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5) * vec2<f32>(1024.0, 48.0);
    let ndc = vec2<f32>(px.x / g.size.x * 2.0 - 1.0, 1.0 - px.y / g.size.y * 2.0);
    var o: VsOut;
    o.clip = vec4<f32>(ndc, 0.0, 1.0);
    o.uv = c * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return o;
}

@fragment
fn fs_overlay(v: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, v.uv);
}
"#;

fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        ' ' => [0; 7],
        _ => [0; 7],
    }
}

fn draw_text(buf: &mut [u8], x0: usize, y0: usize, text: &str, scale: usize, color: [u8; 4]) {
    let mut x = x0;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if bits & (1 << (4 - col)) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = x + col * scale + dx;
                            let py = y0 + row * scale + dy;
                            if px < OVERLAY_W as usize && py < OVERLAY_H as usize {
                                let i = (py * OVERLAY_W as usize + px) * 4;
                                buf[i..i + 4].copy_from_slice(&color);
                            }
                        }
                    }
                }
            }
        }
        x += 6 * scale;
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    tex_pipeline: wgpu::RenderPipeline,
    tex_bind: wgpu::BindGroup,
    tex: wgpu::Texture,
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_bind: wgpu::BindGroup,
    overlay_globals: wgpu::Buffer,
    overlay_tex: wgpu::Texture,
    pixels: Vec<u8>,
    zbuf: Vec<f32>,
    overlay_pixels: Vec<u8>,
    tris: Vec<Tri3>,
    ternary: bool,
    tris_count: usize,
    raster_ms: f32,
    cover: usize,
    visible: usize,
    yaw: f32,
    pitch: f32,
    dist: f32,
    dragging: bool,
    fps: f32,
    frame_ms: f32,
    frame_count: u32,
    fps_timer: Instant,
}

impl State {
    fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("config");
        config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&device, &config);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene"),
            size: wgpu::Extent3d {
                width: W as u32,
                height: H as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tex_view = tex.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let tex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(TEX_WGSL.into()),
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tex_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tex-bg"),
            layout: &tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let tex_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tex-layout"),
            bind_group_layouts: &[Some(&tex_bgl)],
            immediate_size: 0,
        });
        let tex_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene"),
            layout: Some(&tex_layout),
            vertex: wgpu::VertexState {
                module: &tex_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &tex_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_WGSL.into()),
        });
        let overlay_globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("og"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay"),
            size: wgpu::Extent3d {
                width: OVERLAY_W,
                height: OVERLAY_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let overlay_view = overlay_tex.create_view(&Default::default());
        let overlay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let overlay_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay-bg"),
            layout: &overlay_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: overlay_globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&overlay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay-layout"),
            bind_group_layouts: &[Some(&overlay_bgl)],
            immediate_size: 0,
        });
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: Some(&overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_overlay"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_overlay"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let ternary = std::env::args().any(|a| a == "--ternary");
        let tris_count = TIERS[1];
        let tris = gen_mushroom(tris_count);

        let mut state = Self {
            window,
            surface,
            device,
            queue,
            config,
            tex_pipeline,
            tex_bind,
            tex,
            overlay_pipeline,
            overlay_bind,
            overlay_globals,
            overlay_tex,
            pixels: vec![0u8; W * H * 4],
            zbuf: vec![1.0f32; W * H],
            overlay_pixels: vec![0u8; (OVERLAY_W * OVERLAY_H * 4) as usize],
            tris,
            ternary,
            tris_count,
            raster_ms: 0.0,
            cover: 0,
            visible: 0,
            yaw: 0.6,
            pitch: 0.35,
            dist: 9.0,
            dragging: false,
            fps: 0.0,
            frame_ms: 0.0,
            frame_count: 0,
            fps_timer: Instant::now(),
        };
        state.update_globals();
        state.update_overlay();
        state
    }

    fn update_globals(&self) {
        let mut g = [0u8; 16];
        g[0..4].copy_from_slice(&(self.config.width as f32).to_le_bytes());
        g[4..8].copy_from_slice(&(self.config.height as f32).to_le_bytes());
        self.queue.write_buffer(&self.overlay_globals, 0, &g);
    }

    fn update_overlay(&mut self) {
        self.overlay_pixels.fill(0);
        let ok = [120u8, 255, 170, 255];
        let dim = [200u8, 220, 255, 255];
        let line1 = format!(
            "{} | FPS {:>6.1} | FRAME {:>6.2} MS | RASTER {:>6.2} MS",
            if self.ternary {
                "MUSHROOM TERNARY"
            } else {
                "MUSHROOM PLAIN  "
            },
            self.fps,
            self.frame_ms,
            self.raster_ms
        );
        let line2 = format!(
            "TRIS {:>6} | VIS {:>6} | COVER {:>8} | YAW {:>4.2} PITCH {:>4.2} | DRAG ROTATE",
            self.tris_count, self.visible, self.cover, self.yaw, self.pitch
        );
        draw_text(&mut self.overlay_pixels, 16, 6, &line1, 2, ok);
        draw_text(&mut self.overlay_pixels, 16, 30, &line2, 1, dim);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.overlay_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.overlay_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(OVERLAY_W * 4),
                rows_per_image: Some(OVERLAY_H),
            },
            wgpu::Extent3d {
                width: OVERLAY_W,
                height: OVERLAY_H,
                depth_or_array_layers: 1,
            },
        );
    }

    fn render(&mut self) {
        let frame_start = Instant::now();
        self.update_globals();

        // 相机
        let eye = [
            self.dist * self.pitch.cos() * self.yaw.sin(),
            self.dist * self.pitch.sin(),
            self.dist * self.pitch.cos() * self.yaw.cos(),
        ];
        let proj = perspective(60.0f32.to_radians(), W as f32 / H as f32, NEAR, FAR);
        let view = look_at(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = mul(proj, view);

        // CPU 光栅化
        self.pixels.fill(0);
        for px in self.pixels.chunks_exact_mut(4) {
            px[0] = 8;
            px[1] = 10;
            px[2] = 18;
            px[3] = 255;
        }
        self.zbuf.fill(1.0);

        let raster_start = Instant::now();
        let screen = project(&self.tris, vp);
        self.visible = screen.len();
        self.cover = if self.ternary {
            raster_ternary(&mut self.pixels, &mut self.zbuf, &screen)
        } else {
            raster_plain(&mut self.pixels, &mut self.zbuf, &screen)
        };
        self.raster_ms = raster_start.elapsed().as_secs_f32() * 1000.0;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((W * 4) as u32),
                rows_per_image: Some(H as u32),
            },
            wgpu::Extent3d {
                width: W as u32,
                height: H as u32,
                depth_or_array_layers: 1,
            },
        );

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => return,
        };
        let view_tex = output.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_tex,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.tex_pipeline);
            pass.set_bind_group(0, &self.tex_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_tex,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.overlay_pipeline);
            pass.set_bind_group(0, &self.overlay_bind, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([enc.finish()]);
        self.queue.present(output);

        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
            self.frame_count = 0;
            self.fps_timer = Instant::now();
            self.window.set_title(&format!(
                "{} | FPS {:.1} | raster {:.2} ms | tris {}",
                if self.ternary { "MUSHROOM TERNARY" } else { "MUSHROOM PLAIN" },
                self.fps,
                self.raster_ms,
                self.tris_count
            ));
            self.update_overlay();
        }
    }
}

struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("3D Raster: PLAIN vs TERNARY")
                        .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0)),
                )
                .expect("window"),
        );
        self.state = Some(State::new(window));
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let Some(state) = self.state.as_mut() {
            if let DeviceEvent::MouseMotion { delta } = event {
                if state.dragging {
                    state.yaw -= delta.0 as f32 * 0.008;
                    state.pitch = (state.pitch + delta.1 as f32 * 0.008).clamp(-1.45, 1.45);
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    state.config.width = size.width;
                    state.config.height = size.height;
                    state.surface.configure(&state.device, &state.config);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => state.dragging = true,
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => state.dragging = false,
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                state.dist = (state.dist - d * 0.4).clamp(2.0, 20.0);
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ch),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(i) = match ch.as_str() {
                    "1" => Some(0),
                    "2" => Some(1),
                    "3" => Some(2),
                    "4" => Some(3),
                    _ => None,
                } {
                    state.tris_count = TIERS[i];
                    state.tris = gen_mushroom(state.tris_count);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("run");
}
