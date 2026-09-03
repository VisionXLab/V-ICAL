#!/usr/bin/env python3
"""
Video-CL Web UI 启动器（子命令 `serve`）。

启动根目录的 webui:app（FastAPI），默认端口 8888、开启 reload。
"""
import argparse

import uvicorn

from tools._paths import ROOT  # noqa: F401  确保仓库根在 sys.path，uvicorn reload 能 import webui


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vcl serve", description="启动 Web UI 服务器")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    uvicorn.run(
        "webui:app",
        host=args.host,
        port=args.port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
