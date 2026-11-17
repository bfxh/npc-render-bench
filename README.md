# npc-render-bench

大规模 NPC 实时渲染基准：同一套 `wgpu` 管线下的 **2D 精灵** 与 **3D 人形**，可运行的窗口程序，画面内实时显示 FPS / 帧时间 / CPU 更新耗时 / 实例数 / 三角形数。

四个可执行文件：

| bin | 内容 |
|---|---|
| `npc2d` | 10 万个 2D 精灵（人形轮廓），两种更新策略对比 |
| `npc3d` | 12000 个 3D 人形（真实 OBJ 模型，5004 面/个，6000 万三角形/帧） |
| `bench` | 离屏渲染基准：清屏 / 视频单 quad / 10 万实例，纯 GPU 时间 |
| `video` | 把 3D 场景渲染成 H.264 视频（渲染→读回→ffmpeg 管道） |

## 实测数据

环境：Windows · RTX 4060 Ti 8GB · Vulkan · `wgpu` 30 · `winit` 0.30（release 构建，关 vsync）

### 2D 精灵 × 100,000（1600×900，覆盖 102.4 Mpx ≈ 49× 屏幕）

| 模式 | FPS | 帧时间 |
|---|---:|---:|
| GPU-INSTANCED（实例化，位置在 shader 里） | **2245.3** | **0.37 ms** |
| CPU-NAIVE（每帧 CPU 逐条更新 + 上传 2 MB） | **133.1** | **7.47 ms** |

差 **16.9×**。

### 3D 人形 × 12,000（5004 面/个 = 6002.4 万三角形/帧，1600×900）

| 模式 | FPS | 帧时间 |
|---|---:|---:|
| GPU-INSTANCED 单独跑 | **237.2** | 4.00 ms |
| CPU-NAIVE 单独跑 | **238.3** | 4.27 ms |
| 两个窗口并排（GPU 分摊） | 107.6 / 107.0 | 9.00 / 9.84 ms |

**两者几乎相同**——3D 重负载下 GPU 光栅化成为瓶颈（4 ms/帧），CPU 侧那点差异被完全淹没。

### 离屏渲染基准（`bench`，1920×1080）

| 场景 | 最小 ms/帧 | 像素覆盖 |
|---|---:|---:|
| `clear`（基线） | 0.068 | 0 |
| `video 1 quad`（采样 1080p 纹理） | 0.094 | 2.07 Mpx（1× 屏幕） |
| `npc 100k`（10 万个 32×32 实例） | 1.106 | 102.4 Mpx（49.4× 屏幕） |

扣清屏基线后的纯绘制：video 0.026 ms、npc 1.038 ms = **40×**，与像素覆盖比吻合（填充率约 98 Gpx/s）。

### 视频渲染（`video`）

1920×1080 · 30fps · 300 帧，含渲染 + 8.3 MB/帧读回 + H.264 编码：**5.15 ms/帧**。

## 结论

1. **2D 精灵**：批量实例化 vs 逐条更新差 **17×**——瓶颈在 CPU 与每帧上传。
2. **3D 人形**：同一对比几乎为 0——瓶颈转移到 GPU 光栅化（6000 万三角形）。
3. **优化方向由瓶颈决定**：CPU 侧批量化只在 CPU 是瓶颈时有效；3D 重负载要提速得砍面数 / LOD / 剔除。

## 构建与运行

```bash
# 3D 人形需要模型（不随仓库分发，第三方素材）
scripts/fetch_model.sh

cargo build --release
./target/release/npc3d            # 3D，GPU 模式
./target/release/npc3d --cpu      # 3D，CPU 逐条更新模式
./target/release/npc2d            # 2D，GPU 模式
./target/release/npc2d --cpu      # 2D，CPU 逐条更新模式
./target/release/bench            # 离屏渲染基准
./target/release/video out.mp4    # 渲染 3D 场景视频
```

模型查找顺序：`$NPC_MODEL` → exe 同目录 `male02.obj` → 当前目录 → `assets/male02.obj`。

### 操作（窗口程序）

- `1` / `2` / `3` / `4`：切换实例数档位
  - `npc2d`：1 万 / 2.5 万 / 5 万 / 10 万
  - `npc3d`：3 千 / 6 千 / 1.2 万 / 2.4 万
- `Esc`：退出

### 画面参数

窗口左上角与窗口标题同时显示：

```
MODE 3D GPU-INST | FPS 237.2 | FRAME 4.00 MS | CPU 0.00 MS
NPC 12000 | TRIS 60024000 | <GPU 名> | 1..4 TIER | ESC
```

## 结构

```
src/bin/npc2d.rs    2D 精灵窗口（10 万实例，两种更新策略）
src/bin/npc3d.rs    3D 人形窗口（OBJ 加载 + 深度缓冲 + 透视 + 两种更新策略）
src/bin/bench.rs    离屏渲染基准
src/bin/video.rs    3D 场景 → H.264 视频
scripts/fetch_model.sh  下载 male02.obj
```

## 说明

- 两套窗口都关掉了垂直同步（`PresentMode::AutoNoVsync`），FPS 反映的是真实上限。
- 窗口被完全遮挡时 Windows 会节流呈现，FPS 会掉到个位数——这不是性能问题，并排铺开即可恢复。
- `male02.obj` 来自 three.js 示例模型集，仅作本地测试用，未随仓库分发。
