//! 十万 NPC 实时窗口演示（可运行程序，不是视频）。
//!
//! - `winit` 窗口 + `wgpu` 直接渲染，窗口标题与画面左上角同时显示实时参数
//! - 每个 NPC 用 shader 画成**人形**（头 + 躯干 + 两条腿），不是方块
//! - 散点分布 + 缓慢漂移，每个 NPC 尺寸独立
//! - `1..5` 切换实例数（1 万 / 2.5 万 / 5 万 / 10 万 档），`Esc` 退出
//!
//! 运行：`cargo run --release --example npc_window`

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const MAX_NPCS: u32 = 100_000;
const TIERS: [u32; 4] = [10_000, 25_000, 50_000, 100_000];
const OVERLAY_W: u32 = 1024;
const OVERLAY_H: u32 = 48;

// ---------- 人形 shader ----------

const NPC_WGSL: &str = r#"
struct Globals {
    size: vec2<f32>,      // 屏幕像素
    time: f32,
    cpu_mode: f32,        // 1 = CPU 已算好位置，shader 不再叠漂移
};
@group(0) @binding(0) var<uniform> g: Globals;

struct InstIn {
    @location(0) pos: vec2<f32>,   // NDC
    @location(1) scale: vec2<f32>, // NDC 半宽/半高
    @location(2) hue: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) hue: f32,
};

@vertex
fn vs_npc(@builtin(vertex_index) vi: u32, inst: InstIn) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    var o: VsOut;
    o.clip = vec4<f32>(inst.pos + c * inst.scale, 0.0, 1.0);
    o.uv = c * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5); // 0..1，y 向下
    o.hue = inst.hue;
    return o;
}

fn box_(p: vec2<f32>, c: vec2<f32>, h: vec2<f32>) -> f32 {
    let d = abs(p - c) - h;
    return step(max(d.x, d.y), 0.0);
}

@fragment
fn fs_npc(v: VsOut) -> @location(0) vec4<f32> {
    let p = v.uv;
    // 头：圆
    let head_c = vec2<f32>(0.5, 0.17);
    let head = step(length(p - head_c), 0.13);
    // 躯干：上宽下窄的矩形（用两段近似）
    let torso = box_(p, vec2<f32>(0.5, 0.45), vec2<f32>(0.17, 0.16));
    let hips  = box_(p, vec2<f32>(0.5, 0.62), vec2<f32>(0.13, 0.05));
    // 腿
    let leg_l = box_(p, vec2<f32>(0.42, 0.83), vec2<f32>(0.055, 0.16));
    let leg_r = box_(p, vec2<f32>(0.58, 0.83), vec2<f32>(0.055, 0.16));

    let body = max(max(torso, hips), max(leg_l, leg_r));
    let shape = max(head, body);
    if (shape < 0.5) { discard; }

    // 冷色渐变：head 更亮，身体按 hue 变化
    let base = vec3<f32>(0.30 + 0.55 * v.hue, 0.82, 1.0 - 0.45 * v.hue);
    let tint = mix(base, vec3<f32>(1.0, 1.0, 1.0), head * 0.45);
    return vec4<f32>(tint, 1.0);
}
"#;

// ---------- overlay（参数面板）----------

const OVERLAY_WGSL: &str = r#"
struct Globals {
    size: vec2<f32>,
    time: f32,
    _pad: f32,
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
    // 左上角 1:1 像素映射
    let px = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5) * vec2<f32>(f32(1024), f32(48));
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

// ---------- 5x7 点阵字体 ----------

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

/// 在 RGBA 缓冲上写一行点阵文字（scale 倍放大）
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

// ---------- 状态 ----------

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    npc_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    npc_bind: wgpu::BindGroup,
    overlay_tex: wgpu::Texture,
    overlay_bind: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    overlay_pixels: Vec<u8>,
    npcs: u32,
    frame_count: u32,
    fps_timer: Instant,
    fps: f32,
    frame_ms: f32,
    start: Instant,
    gpu_name: String,
    /// true = 朴素模式：每帧在 CPU 上逐条更新十万个 NPC 的位置并整块上传
    cpu_mode: bool,
    /// 原始位置（朴素模式每帧从它重新算，模拟"逐实体状态"）
    base: Vec<u8>,
    /// 上一帧 CPU 更新耗时
    cpu_ms: f32,
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
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface config");
        config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&device, &config);

        let npc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("npc"),
            source: wgpu::ShaderSource::Wgsl(NPC_WGSL.into()),
        });
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_WGSL.into()),
        });

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let npc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("npc-bgl"),
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
        let npc_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("npc-bg"),
            layout: &npc_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });

        let npc_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("npc-layout"),
            bind_group_layouts: &[Some(&npc_bgl)],
            immediate_size: 0,
        });
        let npc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("npc"),
            layout: Some(&npc_layout),
            vertex: wgpu::VertexState {
                module: &npc_shader,
                entry_point: Some("vs_npc"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 20, // pos(8) + scale(8) + hue(4)
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 16,
                            shader_location: 2,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &npc_shader,
                entry_point: Some("fs_npc"),
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

        // overlay 纹理
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let overlay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                    resource: globals.as_entire_binding(),
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

        // 实例缓冲：10 万个精灵的初始位置
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut bytes = Vec::with_capacity(MAX_NPCS as usize * 20);
        for _ in 0..MAX_NPCS {
            let x = (next() % 20_001) as f32 / 10_000.0 - 1.0;
            let y = (next() % 20_001) as f32 / 10_000.0 - 1.0;
            // 人物尺寸：比方块大，一眼能看出是"人"
            let s = 0.010 + (next() % 1_000) as f32 / 1_000.0 * 0.010;
            let hue = (next() % 1_000) as f32 / 1_000.0;
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            bytes.extend_from_slice(&(s * 0.55).to_le_bytes());
            bytes.extend_from_slice(&(s * 2.2).to_le_bytes()); // 人形上下更长
            bytes.extend_from_slice(&hue.to_le_bytes());
        }
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buffer, 0, &bytes);

        let mut state = Self {
            window,
            surface,
            device,
            queue,
            config,
            npc_pipeline,
            overlay_pipeline,
            globals,
            npc_bind,
            overlay_tex,
            overlay_bind,
            instance_buffer,
            overlay_pixels: vec![0u8; (OVERLAY_W * OVERLAY_H * 4) as usize],
            npcs: TIERS[3],
            frame_count: 0,
            fps_timer: Instant::now(),
            fps: 0.0,
            frame_ms: 0.0,
            start: Instant::now(),
            gpu_name,
            cpu_mode: std::env::args().any(|a| a == "--cpu"),
            base: bytes.clone(),
            cpu_ms: 0.0,
        };
        state.update_globals();
        state.update_overlay();
        state
    }

    fn update_globals(&self) {
        let mut data = [0u8; 16];
        data[0..4].copy_from_slice(&(self.config.width as f32).to_le_bytes());
        data[4..8].copy_from_slice(&(self.config.height as f32).to_le_bytes());
        data[8..12].copy_from_slice(&(self.start.elapsed().as_secs_f32()).to_le_bytes());
        self.queue.write_buffer(&self.globals, 0, &data);
    }

    fn update_overlay(&mut self) {
        self.overlay_pixels.fill(0);
        let ok = [120u8, 255, 170, 255];
        let dim = [200u8, 220, 255, 255];
        let line1 = format!(
            "{} | FPS {:>5.1} | FRAME {:>5.2} MS | CPU {:>5.2} MS | NPC {:>6}",
            if self.cpu_mode {
                "MODE CPU-NAIVE "
            } else {
                "MODE GPU-INSTANCED"
            },
            self.fps,
            self.frame_ms,
            self.cpu_ms,
            self.npcs
        );
        let line2 = format!("GPU {} | 1..5 TIER | ESC EXIT", self.gpu_name);
        draw_text(&mut self.overlay_pixels, 16, 8, &line1, 2, ok);
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

        // 朴素模式：每帧在 CPU 上逐条更新十万个 NPC，再整块上传。
        // 这正是"原来的做法"的典型代价：逐实体遍历 + 全量上传。
        if self.cpu_mode {
            let t = self.start.elapsed().as_secs_f32();
            let cpu_start = Instant::now();
            let mut bytes = Vec::with_capacity(self.npcs as usize * 20);
            for i in 0..self.npcs as usize {
                let b = &self.base[i * 20..i * 20 + 20];
                let x = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let y = f32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let phase = x * 3.7 + y * 2.3;
                let dx = (t * 0.9 + phase).sin() * 0.04;
                let dy = (t * 1.3 + phase).cos() * 0.04;
                bytes.extend_from_slice(&(x + dx).to_le_bytes());
                bytes.extend_from_slice(&(y + dy).to_le_bytes());
                bytes.extend_from_slice(&b[8..20]);
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
                            r: 0.015,
                            g: 0.02,
                            b: 0.04,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.npc_pipeline);
            pass.set_bind_group(0, &self.npc_bind, &[]);
            pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            pass.draw(0..6, 0..self.npcs);

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
                "NPC {} | FPS {:.1} | {:.2} ms",
                self.npcs, self.fps, self.frame_ms
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
                        .with_title("十万 NPC 实时演示")
                        .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0)),
                )
                .expect("window"),
        );
        self.state = Some(State::new(window));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
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
                let tier = match ch.as_str() {
                    "1" => Some(0),
                    "2" => Some(1),
                    "3" => Some(2),
                    "4" | "5" => Some(3),
                    _ => None,
                };
                if let Some(i) = tier {
                    state.npcs = TIERS[i];
                    state.window.set_title(&format!("NPC {} | ...", state.npcs));
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
