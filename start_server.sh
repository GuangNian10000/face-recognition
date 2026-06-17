#!/usr/bin/env bash
set -Eeuo pipefail

# 1. 获取脚本所在绝对目录并进入（确保业务代码里的相对路径不报错）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 2. 直接使用原生环境的 python3 启动服务
# 使用 exec 可以让 python 进程直接接管当前 shell，这样 systemd 管理状态会更精准
echo "Starting Face Server using system Python 3..."
exec python3 server_stream.py "$@"