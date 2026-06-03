# LoRA Fine-Tuning with MLflow Experiment Tracking

QLoRA fine-tuning of **Qwen2.5-3B-Instruct** on a Text-to-SQL task using [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl), with full experiment tracking via **MLflow**.

---

## What this does

1. Evaluates the **base model** accuracy on a held-out SQL test set
2. Fine-tunes with **LoRA** (rank 16, ~0.1 % of parameters, ~15–20 min on a single GPU)
3. Re-evaluates the **fine-tuned model** on the same test set
4. Logs everything to MLflow — params, per-step losses, evaluation metrics, plots, dataset sample, and the LoRA adapter weights

| | Base Model | Fine-Tuned |
|---|---|---|
| Task | Text-to-SQL | Text-to-SQL |
| Dataset | `b-mc2/sql-create-context` | same |
| Trainable params | 3 B (frozen) | ~3 M (LoRA only) |

---

## Project layout

```
training_with_mlops/
├── requirements.txt
├── README.md
└── training/
    ├── __init__.py          # package re-exports
    ├── config.py            # dataclass configs (Model, LoRA, Data, Training, MLflow)
    ├── data_loader.py       # DatasetLoader  — download, split, prompt formatting
    ├── model_loader.py      # ModelLoader    — 4-bit load + LoRA adapter application
    ├── evaluator.py         # ModelEvaluator — exact-match SQL evaluation loop
    ├── trainer.py           # LoRATrainer    — SFTTrainer wrapper
    ├── visualizer.py        # Visualizer     — loss curve, accuracy bars, summary chart
    ├── mlflow_logger.py     # MLflowLogger   — all MLflow interactions
    └── main.py              # entry point — orchestrates every step
```

---

## Requirements

- Python 3.11+
- CUDA-capable GPU with at least **16 GB VRAM** (tested on a single A100/H100)
- A running MLflow tracking server (default: `http://192.168.1.50:5007`)

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** `unsloth` bundles a custom CUDA kernel build. If the wheel for your CUDA version is not available on PyPI, follow the [Unsloth installation guide](https://github.com/unslothai/unsloth#installation).

---

## Configuration

All hyperparameters live in `training/config.py` as plain dataclasses — no YAML or CLI flags required. Edit the defaults directly before running:

| Dataclass | Key fields |
|---|---|
| `ModelConfig` | `name`, `max_seq_length`, `load_in_4bit` |
| `LoRAConfig` | `r`, `lora_alpha`, `lora_dropout`, `target_modules` |
| `DataConfig` | `dataset_name`, `num_train`, `num_test`, `seed` |
| `TrainingConfig` | `num_train_epochs`, `learning_rate`, `per_device_train_batch_size` |
| `MLflowConfig` | `tracking_uri`, `experiment_name`, `run_name` |

### Change the MLflow server

```python
# training/config.py
@dataclass
class MLflowConfig:
    tracking_uri: str = "http://192.168.1.50:5007"   # ← update here
    experiment_name: str = "LoRA_SQL_Finetuning"
```

---

## Running

```bash
# from the training_with_mlops/ directory
python -m training.main
```

The script will:

1. Connect to MLflow and create the experiment `LoRA_SQL_Finetuning` (if it does not exist)
2. Open a timestamped run (e.g. `lora_sql_1748860000`)
3. Log all configs as MLflow **parameters**
4. Load and evaluate the base model, then apply LoRA and train
5. Log per-step losses, final metrics, and three PNG plots as **artifacts**
6. Save the LoRA adapter weights and upload them to MLflow
7. Print a final accuracy comparison table and the MLflow run URL

---

## MLflow tracking

Navigate to `http://192.168.1.50:5007` to browse experiments and runs.

### Parameters logged

| Prefix | Examples |
|---|---|
| `model_*` | `model_name`, `model_max_seq_length`, `model_load_in_4bit` |
| `lora_*` | `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules` |
| `training_*` | `training_num_epochs`, `training_learning_rate`, `training_effective_batch_size` |
| `data_*` | `data_dataset_name`, `data_num_train`, `data_num_test` |

### Metrics logged

| Metric | Description |
|---|---|
| `step_train_loss` | Per-step training loss (time series) |
| `train_loss_final` | Final training loss |
| `train_runtime_seconds` | Wall-clock training time |
| `eval_base_accuracy` | Exact-match accuracy before fine-tuning |
| `eval_finetuned_accuracy` | Exact-match accuracy after fine-tuning |
| `eval_accuracy_improvement` | Absolute percentage-point gain |
| `model_trainable_parameters` | Number of LoRA trainable parameters |
| `model_trainable_percentage` | Trainable % of total parameters |

### Artifacts logged

```
mlflow run/
├── plots/
│   ├── training_loss.png        # loss curve over training steps
│   ├── accuracy_comparison.png  # before vs after bar chart
│   └── summary.png              # 3-panel: accuracy + param pie + training time
├── dataset_info/
│   └── dataset_info.json        # sizes, column names, 5 sample rows
└── lora_adapter/
    ├── adapter_config.json      # PEFT adapter configuration
    ├── adapter_model.safetensors
    └── tokenizer files
```

---

## Loading the saved adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct", load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, "/tmp/lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("/tmp/lora_adapter")
```

Or download directly from MLflow:

```python
import mlflow

mlflow.set_tracking_uri("http://192.168.1.50:5007")
mlflow.artifacts.download_artifacts(
    run_id="<run_id>",
    artifact_path="lora_adapter",
    dst_path="./downloaded_adapter",
)
```

---

## Dataset

[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — ~78 K examples from WikiSQL and Spider, each containing:

- `question` — natural language question
- `context` — `CREATE TABLE` schema
- `answer` — expected SQL query

By default, 5 000 examples are used for training and 500 for evaluation.
