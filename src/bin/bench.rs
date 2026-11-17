//! 渲染侧强度对比：十万 NPC 的实例化绘制 vs 视频单纹理合成。
//!
//! | 场景 | 内容 | 像素覆盖 |
//! |---|---|---|
//! | `clear` | 只清屏（基线） | 0 |
//! | `video` | 1 个全屏 quad，采样 1920×1080 纹理（视频帧当纹理） | 1× 屏幕 |
//! | `npc`   | 100,000 个 32×32 实例化 quad（每个 NPC 一个 sprite） | ≈49× 屏幕 |
//!
//! NPC 侧是**全可见上界**：真实引擎会做视锥/遮挡剔除，屏幕上不会同时出现十万个
//! sprite。这里故意不剔除，量的是 GPU 光栅化这一层的天花板。
//!
//! 计时：每帧 `submit + poll(Wait)` 的 wall，预热 5 帧后取 60 帧的最小/平均。

use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const NPCS: u32 = 100_000;

const WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    var o: VsOut;
    o.pos = vec4<f32>(c, 0.0, 1.0);
    o.uv = c * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return o;
}

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
    return vec4<f32>(i.offset + c * i.size, 0.0, 1.0);
}

@group(0) @binding(0) var video: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_video(v: VsOut) -> @location(0) vec4<f32> {
    return textureSample(video, samp, v.uv);
}

@fragment
fn fs_npc() -> @location(0) vec4<f32> {
    return vec4<f32>(0.9, 0.45, 0.2, 1.0);
}
"#;

fn wait(device: &wgpu::Device) {
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}

fn bench_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    mut encode: impl FnMut(&mut wgpu::CommandEncoder),
) -> (f64, f64) {
    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&Default::default());
        encode(&mut enc);
        queue.submit([enc.finish()]);
        wait(device);
    }
    let mut best = f64::INFINITY;
    let mut total = 0.0;
    for _ in 0..60 {
        let started = Instant::now();
        let mut enc = device.create_command_encoder(&Default::default());
        encode(&mut enc);
        queue.submit([enc.finish()]);
        wait(device);
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        best = best.min(ms);
        total += ms;
    }
    println!("{:>12} | {:>8.3} ms | {:>8.3} ms", label, best, total / 60.0);
    (best, total / 60.0)
}

fn color_target() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8Unorm,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    }
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("wgpu 适配器");
    let info = adapter.get_info();
    println!("适配器：{} ({:?})", info.name, info.backend);
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("npc-vs-video"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });

    // 视频纹理：一帧 1080p
    let video_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("video"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let video_view = video_tex.create_view(&Default::default());
    let pixels = vec![128u8; (WIDTH * HEIGHT * 4) as usize];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &video_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(WIDTH * 4),
            rows_per_image: Some(HEIGHT),
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("video-bgl"),
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
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("video-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&video_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let video_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("video-layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let empty_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("npc-layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let video_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("video"),
        layout: Some(&video_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_full"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_video"),
            compilation_options: Default::default(),
            targets: &[Some(color_target())],
        }),
        multiview_mask: None,
        cache: None,
    });

    let npc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("npc"),
        layout: Some(&empty_layout),
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
            targets: &[Some(color_target())],
        }),
        multiview_mask: None,
        cache: None,
    });

    // 十万个 sprite：随机铺满屏幕，每个 32×32 像素
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

    println!("\n{:>12} | {:>11} | {:>11}", "场景", "最小 ms/帧", "平均 ms/帧");
    println!("{:->12}-+-{:->11}-+-{:->11}", "", "", "");

    bench_frame(&device, &queue, "clear", |enc| {
        let _ = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
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
    });

    bench_frame(&device, &queue, "video 1 quad", |enc| {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("video"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
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
        pass.set_pipeline(&video_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    });

    bench_frame(&device, &queue, "npc 100k", |enc| {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("npc"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
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
        pass.set_pipeline(&npc_pipeline);
        pass.set_vertex_buffer(0, instance_buffer.slice(..));
        pass.draw(0..6, 0..NPCS);
    });

    let covered = NPCS as f64 * 32.0 * 32.0;
    println!(
        "\n像素覆盖：video 1× 屏幕 = {:.2} Mpx · npc {:.1}× 屏幕 = {:.1} Mpx",
        WIDTH as f64 * HEIGHT as f64 / 1e6,
        covered / (WIDTH as f64 * HEIGHT as f64),
        covered / 1e6
    );
}
