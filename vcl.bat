@echo off
REM video_cl 统一命令行入口 (Windows)。把全部参数透传给 vcl.py。
REM   vcl serve --port 8888    /    vcl eval-batch ...    /    vcl  (进入菜单)
setlocal
cd /d "%~dp0"
python "%~dp0vcl.py" %*
endlocal
