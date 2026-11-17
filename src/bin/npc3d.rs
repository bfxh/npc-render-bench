//! 3D 人形 × 12000 实时窗口（可运行程序）。
//!
//! 每个 NPC 是一个真正的 3D 人形网格（头 / 躯干 / 双臂 / 双腿，6 个盒子拼成），
//! 带法线与 Lambert 光照，透视相机 + 深度缓冲，散点分布在 3D 空间里。
//!
//! 两种模式（同一个 exe，两个窗口并排看）：
//! - 默认：**GPU-INSTANCED**，位置/旋转全在 shader 里算，per-frame 上传 0
//! - `--cpu`：**CPU-NAIVE**，每帧在 CPU 上逐条算 12000 个 NPC 的旋转并整块上传
//!
//! 画面左上角与窗口标题实时显示：FPS / 帧时间 / CPU 更新耗时 / NP 数 / 三角形数。
//! 按键：`1..4` 切换 3k / 6k / 12k / 24k，`Esc` 退出。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const MAX_NPCS: u32 = 24_000;
const TIERS: [u32; 4] = [3_000, 6_000, 12_000, 24_000];
const OVERLAY_W: u32 = 1024;
const OVERLAY_H: u32 = 48;

// ---------- 3D 人形网格（6 个盒子） ----------

/// 一个轴对齐盒子：中心 c、半尺寸 h，生成 24 顶点（每面独立法线）+ 36 索引
fn push_box(
    verts: &mut Vec<[f32; 6]>,
    idx: &mut Vec<u16>,
    c: [f32; 3],
    h: [f32; 3],
) {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [[h[0], -h[1], -h[2]], [h[0], h[1], -h[2]], [h[0], h[1], h[2]], [h[0], -h[1], h[2]]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-h[0], -h[1], h[2]], [-h[0], h[1], h[2]], [-h[0], h[1], -h[2]], [-h[0], -h[1], -h[2]]]),
        // +Y
        ([0.0, 1.0, 0.0], [[-h[0], h[1], -h[2]], [h[0], h[1], -h[2]], [h[0], h[1], h[2]], [-h[0], h[1], h[2]]]),
        // -Y
        ([0.0, -1.0, 0.0], [[-h[0], -h[1], h[2]], [h[0], -h[1], h[2]], [h[0], -h[1], -h[2]], [-h[0], -h[1], -h[2]]]),
        // +Z
        ([0.0, 0.0, 1.0], [[-h[0], -h[1], h[2]], [h[0], -h[1], h[2]], [h[0], h[1], h[2]], [-h[0], h[1], h[2]]]),
        // -Z
        ([0.0, 0.0, -1.0], [[h[0], -h[1], -h[2]], [-h[0], -h[1], -h[2]], [-h[0], h[1], -h[2]], [h[0], h[1], -h[2]]]),
    ];
    for (normal, corners) in faces {
        let base = verts.len() as u16;
        for p in corners {
            verts.push([
                c[0] + p[0],
                c[1] + p[1],
                c[2] + p[2],
                normal[0],
                normal[1],
                normal[2],
            ]);
        }
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// 一个能认出来的人形：身高约 1.8 个单位
fn humanoid() -> (Vec<[f32; 6]>, Vec<u16>) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    // 头
    push_box(&mut verts, &mut idx, [0.0, 1.62, 0.0], [0.13, 0.15, 0.13]);
    // 脖子
    push_box(&mut verts, &mut idx, [0.0, 1.44, 0.0], [0.05, 0.05, 0.05]);
    // 躯干
    push_box(&mut verts, &mut idx, [0.0, 1.10, 0.0], [0.24, 0.30, 0.13]);
    // 双臂
    push_box(&mut verts, &mut idx, [-0.32, 1.10, 0.0], [0.07, 0.28, 0.07]);
    push_box(&mut verts, &mut idx, [0.32, 1.10, 0.0], [0.07, 0.28, 0.07]);
    // 双腿
    push_box(&mut verts, &mut idx, [-0.12, 0.39, 0.0], [0.09, 0.41, 0.09]);
    push_box(&mut verts, &mut idx, [0.12, 0.39, 0.0], [0.09, 0.41, 0.09]);
    (verts, idx)
}

/// 加载 OBJ 人形模型并归一化：水平居中、脚底 y=0、身高 1.8。
fn load_obj() -> (Vec<[f32; 6]>, Vec<u16>) {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("NPC_MODEL") {
        candidates.push(std::path::PathBuf::from(p));
    }
    if let Some(d) = exe_dir {
        candidates.push(d.join("male02.obj"));
    }
    candidates.push(std::path::PathBuf::from("male02.obj"));
    candidates.push(std::path::PathBuf::from("assets/male02.obj"));
    let path = candidates
        .iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| {
            panic!(
                "找不到 male02.obj；先跑 scripts/fetch_model.sh，或用 NPC_MODEL=<路径> 指定"
            )
        });
    let text = std::fs::read_to_string(path).expect("读 OBJ");

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut verts: Vec<[f32; 6]> = Vec::new();
    let mut idx: Vec<u16> = Vec::new();
    let mut map: HashMap<(usize, usize), u16> = HashMap::new();

    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let x: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                positions.push([x, y, z]);
            }
            Some("vn") => {
                let x: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                normals.push([x, y, z]);
            }
            Some("f") => {
                for token in it {
                    let mut parts = token.split('/');
                    let vi: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let _vt = parts.next();
                    let ni: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    if vi == 0 {
                        continue;
                    }
                    let key = (vi, ni);
                    let id = if let Some(&id) = map.get(&key) {
                        id
                    } else {
                        let p = positions[vi - 1];
                        let n = if ni > 0 && ni <= normals.len() {
                            normals[ni - 1]
                        } else {
                            [0.0, 1.0, 0.0]
                        };
                        let id = verts.len() as u16;
                        verts.push([p[0], p[1], p[2], n[0], n[1], n[2]]);
                        map.insert(key, id);
                        id
                    };
                    idx.push(id);
                }
            }
            _ => {}
        }
    }

    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in &verts {
        for k in 0..3 {
            min[k] = min[k].min(v[k]);
            max[k] = max[k].max(v[k]);
        }
    }
    let height = (max[1] - min[1]).max(1e-6);
    let s = 1.8 / height;
    let cx = (min[0] + max[0]) * 0.5;
    let cz = (min[2] + max[2]) * 0.5;
    for v in &mut verts {
        v[0] = (v[0] - cx) * s;
        v[1] = (v[1] - min[1]) * s;
        v[2] = (v[2] - cz) * s;
    }

    assert!(verts.len() < 65_536, "顶点数超出 u16 索引范围");
    println!(
        "模型 {}：{} 顶点 / {} 三角形",
        path.display(),
        verts.len(),
        idx.len() / 3
    );
    (verts, idx)
}

// ---------- 3D shader ----------

const WGSL: &str = r#"
struct MVP {
    view_proj: mat4x4<f32>,
    time: f32,
    mode: f32,      // 0 = GPU 自转，1 = CPU 已算好 rot_y
    _p0: f32,
    _p1: f32,
};
@group(0) @binding(0) var<uniform> mvp: MVP;

struct InstIn {
    @location(2) pos: vec3<f32>,
    @location(3) scale: f32,
    @location(4) rot_y: f32,
    @location(5) hue: f32,
};

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) hue: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, v: VsIn, inst: InstIn) -> VsOut {
    var angle = inst.rot_y;
    if (mvp.mode < 0.5) {
        angle = angle + mvp.time * (0.7 + inst.hue * 0.9);
    }
    let c = cos(angle);
    let s = sin(angle);
    let rot = mat3x3<f32>(
        c, 0.0, -s,
        0.0, 1.0, 0.0,
        s, 0.0, c,
    );
    let world = rot * (v.position * inst.scale) + inst.pos;

    var o: VsOut;
    o.clip = mvp.view_proj * vec4<f32>(world, 1.0);
    o.normal = rot * v.normal;
    o.hue = inst.hue;
    return o;
}

@fragment
fn fs(v: VsOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.45, 0.85, 0.35));
    let ndl = max(dot(normalize(v.normal), light), 0.0);
    let base = vec3<f32>(
        0.30 + 0.55 * v.hue,
        0.72 + 0.15 * (1.0 - v.hue),
        1.0 - 0.40 * v.hue,
    );
    return vec4<f32>(base * (0.22 + 0.78 * ndl), 1.0);
}
"#;

// ---------- overlay ----------

const OVERLAY_WGSL: &str = r#"
struct Globals {
    size: vec2<f32>,
    time: f32,
    _p: f32,
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

// ---------- 矩阵 ----------

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
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
        let l = dot(a, a).sqrt();
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

fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
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

// ---------- 状态 ----------

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    mvp: wgpu::Buffer,
    mvp_bind: wgpu::BindGroup,
    overlay_globals: wgpu::Buffer,
    overlay_tex: wgpu::Texture,
    overlay_bind: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    overlay_pixels: Vec<u8>,
    base: Vec<u8>,
    npcs: u32,
    cpu_mode: bool,
    cpu_ms: f32,
    frame_count: u32,
    fps_timer: Instant,
    fps: f32,
    frame_ms: f32,
    start: Instant,
    gpu_name: String,
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
        let gpu_name = adapter.get_info().name.clone();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));
        let mut config = surface
            .get_default_config(&adapter, w, h)
            .expect("surface config");
        config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("npc3d"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });

        // 真实人形模型：male02.obj（2746 顶点 / 5004 面）
        let (verts, idx) = load_obj();
        let mut vbuf = Vec::with_capacity(verts.len() * 24);
        for v in &verts {
            for f in v {
                vbuf.extend_from_slice(&f.to_le_bytes());
            }
        }
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertices"),
            size: vbuf.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &vbuf);
        let mut ibuf = Vec::with_capacity(idx.len() * 2);
        for i in &idx {
            ibuf.extend_from_slice(&i.to_le_bytes());
        }
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indices"),
            size: ibuf.len() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&index_buffer, 0, &ibuf);
        let index_count = idx.len() as u32;

        // MVP uniform: mat4(64) + time(4) + mode(4) + pad(8) = 80
        let mvp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mvp"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mvp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mvp-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mvp_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mvp-bg"),
            layout: &mvp_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: mvp.as_entire_binding(),
            }],
        });
        let mvp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mvp-layout"),
            bind_group_layouts: &[Some(&mvp_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("npc3d"),
            layout: Some(&mvp_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 24,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                        ],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 24,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 16,
                                shader_location: 4,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 20,
                                shader_location: 5,
                            },
                        ],
                    }),
                ],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        // 深度纹理
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());

        // overlay
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_WGSL.into()),
        });
        let overlay_globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay-globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay-tex"),
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
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

        // 实例：3D 散点 + 随机朝向/尺寸/色调
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut bytes = Vec::with_capacity(MAX_NPCS as usize * 24);
        let extent = 9.0f32;
        for _ in 0..MAX_NPCS {
            let x = (next() % 20_001) as f32 / 10_000.0 * extent - extent * 0.5;
            let z = (next() % 20_001) as f32 / 10_000.0 * extent - extent * 0.5;
            let y = (next() % 2_001) as f32 / 1_000.0; // 0..2
            let scale = 0.7 + (next() % 1_000) as f32 / 1_000.0 * 0.6;
            let rot = (next() % 6_283) as f32 / 1_000.0;
            let hue = (next() % 1_000) as f32 / 1_000.0;
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            bytes.extend_from_slice(&z.to_le_bytes());
            bytes.extend_from_slice(&scale.to_le_bytes());
            bytes.extend_from_slice(&rot.to_le_bytes());
            bytes.extend_from_slice(&hue.to_le_bytes());
        }
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buffer, 0, &bytes);

        let cpu_mode = std::env::args().any(|a| a == "--cpu");
        let mut state = Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            overlay_pipeline,
            mvp,
            mvp_bind,
            overlay_globals,
            overlay_tex,
            overlay_bind,
            vertex_buffer,
            index_buffer,
            index_count,
            instance_buffer,
            depth,
            depth_view,
            overlay_pixels: vec![0u8; (OVERLAY_W * OVERLAY_H * 4) as usize],
            base: bytes.clone(),
            npcs: TIERS[2], // 默认 12000（"一万多"）
            cpu_mode,
            cpu_ms: 0.0,
            frame_count: 0,
            fps_timer: Instant::now(),
            fps: 0.0,
            frame_ms: 0.0,
            start: Instant::now(),
            gpu_name,
        };
        state.update_mvp();
        state.update_overlay();
        state
    }

    fn update_mvp(&self) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = perspective(45.0f32.to_radians(), aspect, 0.1, 100.0);
        let eye = [0.0, 9.0, 15.0];
        let view = look_at(eye, [0.0, 0.8, 0.0], [0.0, 1.0, 0.0]);
        let vp = mul(proj, view);
        let mut data = [0u8; 80];
        for (i, row) in vp.iter().enumerate() {
            for (j, f) in row.iter().enumerate() {
                data[i * 16 + j * 4..i * 16 + j * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
        data[64..68].copy_from_slice(&self.start.elapsed().as_secs_f32().to_le_bytes());
        data[68..72].copy_from_slice(&(if self.cpu_mode { 1.0f32 } else { 0.0 }).to_le_bytes());
        self.queue.write_buffer(&self.mvp, 0, &data);

        let mut og = [0u8; 16];
        og[0..4].copy_from_slice(&(self.config.width as f32).to_le_bytes());
        og[4..8].copy_from_slice(&(self.config.height as f32).to_le_bytes());
        self.queue.write_buffer(&self.overlay_globals, 0, &og);
    }

    fn update_overlay(&mut self) {
        self.overlay_pixels.fill(0);
        let ok = [120u8, 255, 170, 255];
        let dim = [200u8, 220, 255, 255];
        let tris = self.npcs as u64 * (self.index_count as u64 / 3);
        let line1 = format!(
            "{} | FPS {:>6.1} | FRAME {:>5.2} MS | CPU {:>5.2} MS",
            if self.cpu_mode {
                "MODE 3D CPU-NAIVE "
            } else {
                "MODE 3D GPU-INST  "
            },
            self.fps,
            self.frame_ms,
            self.cpu_ms
        );
        let line2 = format!(
            "NPC {:>6} | TRIS {:>9} | {} | 1..4 TIER | ESC",
            self.npcs, tris, self.gpu_name
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
        self.update_mvp();

        // CPU-NAIVE：每帧逐条重算 12000 个 NPC 的旋转并整块上传
        if self.cpu_mode {
            let t = self.start.elapsed().as_secs_f32();
            let cpu_start = Instant::now();
            let n = self.npcs as usize;
            let mut bytes = Vec::with_capacity(n * 24);
            for i in 0..n {
                let b = &self.base[i * 24..i * 24 + 24];
                let rot = f32::from_le_bytes([b[16], b[17], b[18], b[19]]);
                let hue = f32::from_le_bytes([b[20], b[21], b[22], b[23]]);
                let new_rot = rot + t * (0.7 + hue * 0.9);
                bytes.extend_from_slice(&b[0..16]);
                bytes.extend_from_slice(&new_rot.to_le_bytes());
                bytes.extend_from_slice(&b[20..24]);
            }
            self.queue.write_buffer(&self.instance_buffer, 0, &bytes);
            self.cpu_ms = cpu_start.elapsed().as_secs_f32() * 1000.0;
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.04,
                            b: 0.07,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.mvp_bind, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.index_count, 0, 0..self.npcs);
        }
        // overlay 单独一个 pass：无深度附件，才能与无深度的 overlay 管线匹配
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                "3D NPC {} | FPS {:.1} | {:.2} ms",
                self.npcs, self.fps, self.frame_ms
            ));
            self.update_overlay();
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = self.depth.create_view(&Default::default());
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
                        .with_title("3D 人形 × 12000")
                        .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0)),
                )
                .expect("window"),
        );
        self.state = Some(State::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
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
                    state.npcs = TIERS[i];
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
