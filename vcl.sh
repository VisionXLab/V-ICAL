#!/usr/bin/env bash
# video_cl 统一命令行入口 (Unix)。把全部参数透传给 vcl.py。
#   ./vcl.sh serve --port 8888   /   ./vcl.sh eval-batch ...   /   ./vcl.sh  (进入菜单)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
exec python "$DIR/vcl.py" "$@"
