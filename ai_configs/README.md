# AI 配置数据

正式的 `ai_configs/` 内容存放在私有 Hugging Face 仓库中，不随 GitHub 代码仓库发布。

完成 Hugging Face 登录后，在 V-ICAL 根目录执行：

```bash
hf download <你的账号>/V-ICAL-ai-configs --repo-type dataset --local-dir ai_configs
```

下载后，`ai_configs/` 应包含按游戏和配置名称组织的 `config.json`、动作序列及上下文示例文件。
