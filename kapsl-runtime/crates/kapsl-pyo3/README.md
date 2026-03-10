# kapsl-runtime

Python SDK for [kapsl-runtime](https://kapsl.ai) — a high-performance on-device inference engine for LLMs and multimodal models.

## Installation

```bash
pip install kapsl-runtime
```

## Usage

```python
from kapsl_runtime import KapslClient

client = KapslClient()
result = client.infer(model_id=0, inputs={"input_ids": ...})
```

## License

Proprietary — see [kapsl.ai](https://kapsl.ai) for licensing information.
