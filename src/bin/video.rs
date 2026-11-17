//! 把十万 NPC 场景**真的渲染成视频**（不是数字，是能看的画面）。
//!
//! 与 `bench_npc_render` 同一套管线，差别：
//! - target 带 `COPY_SRC`，每帧读回像素
//! - 通过 stdin 管道喂给 ffmpeg 编码成 H.264
//! - NPC 用 shader 里的时间 uniform 做运动（instance buffer 只传一次，模拟引擎每帧更新位置）
//!
//! 输出：1920×1080、30 fps、10 秒 = 300 帧。

use std::io::Write;
use std::process::{Command, Stdio};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const NPCS: u32 = 100_000;
const FRAMES: u32 = 300;

const WGSL: &str = r#"
struct U {
    t: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};
@group(0) @binding(0) var<uniform> u: U;

struct InstIn {
    @location(0) offset: vec2<f32>,
    @location(1) size: f32,
};

@vertex
fn vs_inst(@builtin(vertex_index) vi: u32, i: InstIn) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    // 每只 NPC 有自己的相位：位置随时间漂移，形成群体运动
    let phase = i.offset.x * 3.7 + i.offset.y * 2.3;
    let drift = vec2<f32>(sin(u.t * 0.9 + phase), cos(u.t * 1.3 + phase)) * 0.04;
    return vec4<f32>(i.offset + drift + c * i.size, 0.0, 1.0);
}

@fragment
fn fs_npc() -> @location(0) vec4<f32> {
    // 冷色小方块，在深色背景上清晰可见
    return vec4<f32>(0.35, 0.85, 1.0, 1.0);
}
"#;

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("wgpu 适配器");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("wgpu device");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("npc-video"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });

    // 时间 uniform（16 字节对齐）
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("time"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("time-bgl"),
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
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("time-bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("npc"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_inst"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            })],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_npc"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    // 十万个 NPC 的位置（16 字节/实例）
    let mut rng = 0x243F_6A88_85A3_08D3u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let half = 32.0f32 / WIDTH as f32;
    let mut bytes = Vec::with_capacity(NPCS as usize * 16);
    for _ in 0..NPCS {
        let x = (next() % 20_001) as f32 / 10_000.0 - 1.0;
        let y = (next() % 20_001) as f32 / 10_000.0 - 1.0;
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes.extend_from_slice(&half.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
    }
    let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instances"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&instance_buffer, 0, &bytes);

    // 读回缓冲：1920×4 = 7680，正好 256 对齐
    let bytes_per_row = WIDTH * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * HEIGHT) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "npc_100k_render.mp4".to_string());
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{WIDTH}x{HEIGHT}"),
            "-r",
            "30",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            &out,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("启动 ffmpeg");
    let mut stdin = ffmpeg.stdin.take().expect("ffmpeg stdin");

    println!("渲染 {FRAMES} 帧（{WIDTH}×{HEIGHT} @30fps）→ {out}");
    let started = std::time::Instant::now();
    for frame in 0..FRAMES {
        let t = frame as f32 / 30.0;
        queue.write_buffer(&uniform, 0, &t.to_le_bytes());

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.03,
                            b: 0.06,
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, instance_buffer.slice(..));
            pass.draw(0..6, 0..NPCS);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let data = slice.get_mapped_range().expect("映射读回缓冲");
        stdin.write_all(&data).expect("写 ffmpeg");
        drop(data);
        readback.unmap();

        if frame % 30 == 0 {
            print!("\r  {frame}/{FRAMES}");
            let _ = std::io::stdout().flush();
        }
    }
    drop(stdin);
    let status = ffmpeg.wait().expect("ffmpeg 结束");
    println!(
        "\n完成：{:?}（ffmpeg {:?}），{:.2} 秒/帧",
        std::time::Instant::now().duration_since(started) / FRAMES,
        status.code(),
        started.elapsed().as_secs_f64() / FRAMES as f64 * 1000.0
    );
    println!("输出：{out}");
}
