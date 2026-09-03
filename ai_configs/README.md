# AI Configuration Data

The full `ai_configs/` dataset is stored in the private Hugging Face repository
[`Mingqian-233/V-ICAL-ai-configs`](https://huggingface.co/datasets/Mingqian-233/V-ICAL-ai-configs). It is not committed to the GitHub source repository.

After installing the project dependencies, authenticate with Hugging Face and explicitly download the data into `ai_configs/` from the V-ICAL root directory:

```bash
hf auth login
hf download Mingqian-233/V-ICAL-ai-configs --repo-type dataset --local-dir ai_configs
```

The downloaded directory contains per-game `config.json` files, context frames, videos, and references to preload action sequences used by batch evaluation.
