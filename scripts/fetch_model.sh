#!/usr/bin/env bash
# 下载 three.js 示例里的人形模型 male02.obj（约 454 KB，2746 顶点 / 5004 面）。
# 模型不随仓库分发（第三方素材），需要时本地拉取。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets
url="https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/obj/male02/male02.obj"
echo "下载 $url"
curl -sL --max-time 120 -o assets/male02.obj "$url"
bytes=$(wc -c < assets/male02.obj)
echo "完成：assets/male02.obj（$bytes 字节）"
if [ "$bytes" -lt 10000 ]; then
  echo "文件太小，可能是代理返回的错误页；请手动检查。" >&2
  exit 1
fi
