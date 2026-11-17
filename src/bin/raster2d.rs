//! 桌面程序：同一个三角形场景，两套 CPU 光栅化实现。
//!
//! - 默认 `--plain`：逐像素 edge function（纯 Rust 传统写法）
//! - `--ternary`：tile 三态 + 位掩码（三进制位平面写法）
//!
//! 每帧都在 CPU 上真的把三角形光栅化进 960×540 的像素缓冲，再上传成纹理显示。
//! 画面左上角与窗口标题实时显示：FPS / 帧时间 / 光栅化耗时 / 三角形数 / 覆盖像素。
//!
//! 按键：`1..4` 切换 1 万 / 3 万 / 6 万 / 10 万三角形，`Esc` 退出。

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const W: usize = 960;
const H: usize = 540;
const TILE: usize = 8;
const TW: usize = W / TILE;
const TH: usize = H / TILE;
const TIERS: [usize; 4] = [10_000, 30_000, 60_000, 100_000];
const OVERLAY_W: u32 = 1024;
const OVERLAY_H: u32 = 48;

#[derive(Clone, Copy)]
struct Tri {
    v: [[f32; 2]; 3],
    color: [u8; 4],
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

fn gen_tris(n: usize) -> Vec<Tri> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut tris = Vec::with_capacity(n);
    for i in 0..n {
        let cx = rng.f() * W as f32;
        let cy = rng.f() * H as f32;
        // 一半小三角形（粒子/UI），一半大三角形（地形/墙面）
        let r = if i % 2 == 0 {
            2.0 + rng.f() * 5.0
        } else {
            12.0 + rng.f() * 48.0
        };
        let a0 = rng.f() * std::f32::consts::TAU;
        let a1 = a0 + 2.0 * std::f32::consts::TAU / 3.0;
        let a2 = a1 + 2.0 * std::f32::consts::TAU / 3.0;
        let p = |a: f32| [cx + r * a.cos(), cy + r * a.sin()];
        let mut t = Tri {
            v: [p(a0), p(a1), p(a2)],
            color: [
                (60 + (rng.next() % 160) as u8),
                (120 + (rng.next() % 120) as u8),
                (180 + (rng.next() % 75) as u8),
                255,
            ],
        };
        // 显式逆时针，保证内部三条 edge 同为正
        let cross = (t.v[1][0] - t.v[0][0]) * (t.v[2][1] - t.v[0][1])
            - (t.v[1][1] - t.v[0][1]) * (t.v[2][0] - t.v[0][0]);
        if cross < 0.0 {
            t.v.swap(1, 2);
        }
        tris.push(t);
    }
    tris
}

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
    (
        (x0.floor().max(0.0)) as usize,
        (y0.floor().max(0.0)) as usize,
        ((x1.ceil() as usize) + 1).min(W),
        ((y1.ceil() as usize) + 1).min(H),
    )
}

/// 纯 Rust 写法：逐像素三次 edge function
fn raster_plain(pixels: &mut [u8], tris: &[Tri]) -> usize {
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
                    let i = (y * W + x) * 4;
                    pixels[i..i + 4].copy_from_slice(&tri.color);
                    covered += 1;
                }
            }
        }
    }
    covered
}

/// 三进制写法：tile 三态分类 + 整块结算
fn raster_ternary(pixels: &mut [u8], tris: &[Tri]) -> usize {
    let mut covered = 0usize;
    for tri in tris {
        let e = [
            (tri.v[0], tri.v[1]),
            (tri.v[1], tri.v[2]),
            (tri.v[2], tri.v[0]),
        ];
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
                    // 整块 64 像素一次结算（凸性保证精确）
                    for y in ty * TILE..(ty + 1) * TILE {
                        let row = y * W;
                        for x in tx * TILE..(tx + 1) * TILE {
                            let i = (row + x) * 4;
                            pixels[i..i + 4].copy_from_slice(&tri.color);
                        }
                    }
                    covered += TILE * TILE;
                    continue;
                }
                // 混合 tile：逐像素
                for y in ty * TILE..(ty + 1) * TILE {
                    let py = y as f32 + 0.5;
                    for x in tx * TILE..(tx + 1) * TILE {
                        let p = [x as f32 + 0.5, py];
                        if edge(e[0].0, e[0].1, p) >= 0.0
                            && edge(e[1].0, e[1].1, p) >= 0.0
                            && edge(e[2].0, e[2].1, p) >= 0.0
                        {
                            let i = (y * W + x) * 4;
                            pixels[i..i + 4].copy_from_slice(&tri.color);
                            covered += 1;
                        }
                    }
                }
            }
        }
    }
    covered
}

// ---------- shader ----------

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
    overlay_pixels: Vec<u8>,
    tris: Vec<Tri>,
    ternary: bool,
    tris_count: usize,
    raster_ms: f32,
    cover: usize,
    fps: f32,
    frame_ms: f32,
    frame_count: u32,
    fps_timer: Instant,
    start: Instant,
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

        // 场景纹理
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

        // overlay
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
        let tris = gen_tris(tris_count);

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
            overlay_pixels: vec![0u8; (OVERLAY_W * OVERLAY_H * 4) as usize],
            tris,
            ternary,
            tris_count,
            raster_ms: 0.0,
            cover: 0,
            fps: 0.0,
            frame_ms: 0.0,
            frame_count: 0,
            fps_timer: Instant::now(),
            start: Instant::now(),
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
                "TERNARY TILE+PLANE "
            } else {
                "PLAIN PER-PIXEL   "
            },
            self.fps,
            self.frame_ms,
            self.raster_ms
        );
        let line2 = format!(
            "TRIS {:>6} | COVER {:>8} | {}x{} | 1..4 TIER | ESC",
            self.tris_count, self.cover, W, H
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

        // CPU 光栅化（真实成本，每帧都跑）
        self.pixels.fill(0x0a);
        for px in self.pixels.chunks_exact_mut(4) {
            px[0] = 10;
            px[1] = 12;
            px[2] = 20;
            px[3] = 255;
        }
        let raster_start = Instant::now();
        self.cover = if self.ternary {
            raster_ternary(&mut self.pixels, &self.tris)
        } else {
            raster_plain(&mut self.pixels, &self.tris)
        };
        self.raster_ms = raster_start.elapsed().as_secs_f32() * 1000.0;

        // 上传场景
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
        let view = output.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                "{} | FPS {:.1} | raster {:.2} ms | tris {}",
                if self.ternary {
                    "TERNARY"
                } else {
                    "PLAIN"
                },
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
                        .with_title("Raster: PLAIN vs TERNARY")
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
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    state.config.width = size.width;
                    state.config.height = size.height;
                    state.surface.configure(&state.device, &state.config);
                }
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
                    state.tris = gen_tris(state.tris_count);
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
