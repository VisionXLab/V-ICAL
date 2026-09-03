"""video_cl 命令行功能包。

由根目录唯一入口 `vcl.py` 统一调度。各模块对应一个子命令：

    serve / play / batch / viewer / eval_batch / eval_crossval /
    count_fails / tokens / rule_extract / rule_extract_run / rule_vs_pass /
    export_trajectories

所有模块通过 `from tools._paths import ROOT` 锚定项目根目录，
ROOT 始终指向仓库根（tools/ 的上一级），与脚本被调用的 cwd 无关。
"""
