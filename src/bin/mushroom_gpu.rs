//! 毒蘑菇 GPU 压力测试：同一场景，两套 GPU 内核。
//!
//! - 默认 `--binary`：f32 SDF + sphere tracing（标准距离场步进）
//! - `--ternary`：三值量化（-1/0/+1）后固定步长（丢掉距离信息，只能盲步进）
//!
//! 场景：SDF 拼出的毒蘑菇（压扁球伞盖 + 圆柱菌柄），体积雾 + 红伞白点。
//! 画面左上角与窗口标题实时显示 FPS / 帧时间 / 分辨率 / 模式。

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const OVERLAY_W: u32 = 1024;
const OVERLAY_H: u32 = 48;

const WGSL: &str = r#"
struct U {
    res: vec2<f32>,
    time: f32,
    ternary: f32,
};
@group(0) @binding(0) var<uniform> u: U;

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat2x2<f32>(c, -s, s, c);
}

// ---- SDF：毒蘑菇 ----
fn sd_cap(p: vec3<f32>) -> f32 {
    // 压扁球（伞盖），并稍微上拱
    let q = vec3<f32>(p.x, (p.y - 0.62) * 1.45, p.z);
    return length(q) - 1.0;
}

fn sd_stem(p: vec3<f32>) -> f32 {
    let d = vec2<f32>(length(p.xz) - 0.20, abs(p.y + 0.28) - 0.62);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0))) - 0.02;
}

fn sd_mushroom(p: vec3<f32>) -> f32 {
    return min(sd_cap(p), sd_stem(p));
}

// 伞盖上的白点（用若干小球的并集近似）
fn spots(p: vec3<f32>) -> f32 {
    var d = 1e9;
    for (var i = 0; i < 14; i = i + 1) {
        let f = f32(i);
        let a = f * 2.399963; // 黄金角
        let r = 0.25 + fract(f * 0.37) * 0.62;
        let c = vec3<f32>(cos(a) * r, 0.62 + sqrt(max(0.0, 1.0 - r * r)) * 0.42, sin(a) * r);
        d = min(d, length(p - c) - (0.055 + fract(f * 0.71) * 0.045));
    }
    return d;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy / u.res * 2.0 - vec2<f32>(1.0);
    let ro = vec3<f32>(0.0, 0.85, 3.2);
    var rd = normalize(vec3<f32>(uv.x * 1.15, uv.y * 1.15 - 0.25, -1.6));
    rd = vec3<f32>(rot2(u.time * 0.15) * rd.xy, rd.z);

    var t = 0.0;
    var hit = false;
    var glow = 0.0;

    if (u.ternary > 0.5) {
        // 三进制内核：SDF 量化成 -1 / 0 / +1，丢掉距离 → 只能固定步长盲步进
        for (var i = 0; i < 96; i = i + 1) {
            let p = ro + rd * t;
            let d = sd_mushroom(p);
            // 三值化：-1 内部 / 0 表面 / +1 外部
            let tri = select(select(-1.0, 0.0, abs(d) < 0.03), 1.0, d > 0.0);
            glow += max(0.0, 1.0 - abs(d) * 2.0) * 0.01;
            if (tri <= 0.0) {
                hit = true;
                break;
            }
            t = t + 0.055; // 三值没有距离信息，只能盲步进
            if (t > 8.0) { break; }
        }
    } else {
        // 二进制内核：f32 距离 + sphere tracing
        for (var i = 0; i < 96; i = i + 1) {
            let p = ro + rd * t;
            let d = sd_mushroom(p);
            glow += max(0.0, 1.0 - abs(d) * 2.0) * 0.01;
            if (d < 0.0015) {
                hit = true;
                break;
            }
            t = t + max(d, 0.002);
            if (t > 8.0) { break; }
        }
    }

    var col = vec3<f32>(0.05, 0.05, 0.08) * (1.0 - uv.y * 0.25);

    if (hit) {
        let p = ro + rd * t;
        // 法线
        let e = 0.002;
        let n = normalize(vec3<f32>(
            sd_mushroom(p + vec3<f32>(e, 0.0, 0.0)) - sd_mushroom(p - vec3<f32>(e, 0.0, 0.0)),
            sd_mushroom(p + vec3<f32>(0.0, e, 0.0)) - sd_mushroom(p - vec3<f32>(0.0, e, 0.0)),
            sd_mushroom(p + vec3<f32>(0.0, 0.0, e)) - sd_mushroom(p - vec3<f32>(0.0, 0.0, e)),
        ));
        let lig = normalize(vec3<f32>(0.5, 0.9, 0.4));
        let diff = max(dot(n, lig), 0.0);
        // 伞盖红 / 菌柄米白 / 白点
        var base = vec3<f32>(0.62, 0.06, 0.05);
        if (p.y < 0.55) {
            base = vec3<f32>(0.86, 0.84, 0.78);
        }
        if (spots(p) < 0.0 && p.y > 0.55) {
            base = vec3<f32>(0.97, 0.96, 0.93);
        }
        col = base * (0.18 + 0.9 * diff);
        // 边缘暗化
        col = col * (0.4 + 0.6 * smoothstep(1.0, 0.2, length(uv)));
        // 体积雾
        col = col * exp(-t * 0.12) + vec3<f32>(0.15, 0.05, 0.05) * glow;
    } else {
        col = col + vec3<f32>(0.20, 0.03, 0.02) * glow;
    }

    return vec4<f32>(pow(col, vec3<f32>(0.85)), 1.0);
}
"#;

const OVERLAY_WGSL: &str = r#"
struct G { size: vec2<f32>, _a: f32, _b: f32 };
@group(0) @binding(0) var<uniform> g: G;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };

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

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scene_pipeline: wgpu::RenderPipeline,
    scene_bind: wgpu::BindGroup,
    scene_u: wgpu::Buffer,
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_bind: wgpu::BindGroup,
    overlay_globals: wgpu::Buffer,
    overlay_tex: wgpu::Texture,
    overlay_pixels: Vec<u8>,
    ternary: bool,
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

        let scene_u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-u"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mushroom"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let scene_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene-bg"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_u.as_entire_binding(),
            }],
        });
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene-layout"),
            bind_group_layouts: &[Some(&scene_bgl)],
            immediate_size: 0,
        });
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
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

        let ternary = std::env::args().any(|a| a == "--ternary");
        let mut state = Self {
            window,
            surface,
            device,
            queue,
            config,
            scene_pipeline,
            scene_bind,
            scene_u,
            overlay_pipeline,
            overlay_bind,
            overlay_globals,
            overlay_tex,
            overlay_pixels: vec![0u8; (OVERLAY_W * OVERLAY_H * 4) as usize],
            ternary,
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
        g[8..12].copy_from_slice(&(self.start.elapsed().as_secs_f32()).to_le_bytes());
        g[12..16].copy_from_slice(&(if self.ternary { 1.0f32 } else { 0.0 }).to_le_bytes());
        self.queue.write_buffer(&self.scene_u, 0, &g);

        let mut og = [0u8; 16];
        og[0..4].copy_from_slice(&(self.config.width as f32).to_le_bytes());
        og[4..8].copy_from_slice(&(self.config.height as f32).to_le_bytes());
        self.queue.write_buffer(&self.overlay_globals, 0, &og);
    }

    fn update_overlay(&mut self) {
        self.overlay_pixels.fill(0);
        let ok = [120u8, 255, 170, 255];
        let dim = [200u8, 220, 255, 255];
        let line1 = format!(
            "{} | FPS {:>6.1} | FRAME {:>6.2} MS",
            if self.ternary {
                "TOXIC MUSHROOM / TERNARY"
            } else {
                "TOXIC MUSHROOM / BINARY "
            },
            self.fps,
            self.frame_ms
        );
        let line2 = format!(
            "{}x{} | 96 STEPS RAYMARCH | ESC EXIT",
            self.config.width, self.config.height
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
            pass.set_pipeline(&self.scene_pipeline);
            pass.set_bind_group(0, &self.scene_bind, &[]);
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
                "TOXIC MUSHROOM {} | FPS {:.1} | {:.2} ms",
                if self.ternary { "TERNARY" } else { "BINARY" },
                self.fps,
                self.frame_ms
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
                        .with_title("Toxic Mushroom GPU Test")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0)),
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
