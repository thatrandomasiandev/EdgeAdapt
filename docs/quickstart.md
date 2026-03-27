# Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --extras dev
```

Minimal usage:

```python
import edgeadapt as ea
from edgeadapt.policy import MaximizeAccuracy
from edgeadapt.registry import ModelFamily

family = ModelFamily.from_yaml("family.yaml")
engine = ea.Engine(family, MaximizeAccuracy(latency_ceiling_ms=100.0))
engine.start()
out = engine.infer(input_data)
engine.stop()
```
