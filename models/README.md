# Weight-level self-improved simulators (LoRA adapters)

These are the trained **weight-level** AutoSim generators from RQ2: LoRA adapters
fine-tuned so that the simulator's utterances become indistinguishable from real
iKAT users, driving the real-versus-simulated discriminator from AUC ≈ 0.88 down
to ≈ 0.5. They are the "models" released with the paper.

| Adapter | Base model | Role |
|---|---|---|
| `adapter_qwen2.5-3b`     | `Qwen/Qwen2.5-3B-Instruct`      | primary weight-level simulator |
| `adapter_qwen2.5-7b`     | `Qwen/Qwen2.5-7B-Instruct`      | scale check (larger base model) |
| `adapter_mistral-7b`     | `mistralai/Mistral-7B-Instruct-v0.3` | cross-family check |

Each adapter is LoRA (`r=16`, `α=32`, `q`/`v` projections, `task_type=CAUSAL_LM`)
trained by adversarial rejection-sampling SFT against a held-out real-versus-simulated
discriminator (see `experiments/exp_ikat_weight.py` and `results/weight_runs/`).

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-3B-Instruct"
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto"),
    "models/adapter_qwen2.5-3b",
)
tok = AutoTokenizer.from_pretrained(base_id)
```

Base weights are pulled from the Hugging Face Hub at load time; only the adapters
(14–26 MB each) are stored here.
