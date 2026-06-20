from dataclasses import dataclass, field
from typing import List, Optional
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PromptConfig:
    text: str = (
        "You are a SQL assistant. Given a database schema and a question, "
        "output ONLY the SQL query that answers the question. "
        "Do not include any explanation, markdown formatting, or extra text."
    )
    commit_message: str = "Registered via training run"
    version: str = "latest"


@dataclass
class ModelConfig:
    name: str = "unsloth/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    dtype: Optional[str] = None


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class DataConfig:
    dataset_name: str = "c"
    target_column: str = "answer"
    num_train: int = 5000
    num_test: int = 500
    seed: int = 42


@dataclass
class TrainingConfig:
    output_dir: str = "./lora_sql_checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "no"
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass
class MLflowConfig:
    # Authentication related configs
    tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5007")
    )
    username: Optional[str] = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_USERNAME")
    )
    password: Optional[str] = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_PASSWORD")
    )

    # Project related configs
    project_name: str = "SQL_Finetuning_Project"
    experiment_name: str = "SQL_Finetuning"
    description: str = "Fine tuning qwen model for SQL query prediction."
    run_name: Optional[str] = "sql_finetuning"

    # Prompt registry related configs
    prompt_registry_name: str = "sql-assistant-system-prompt"

    # Model related configs
    model_artifact_path: str = "model_adapter"
    registered_model_name: str = "sql-qwen1.5"

    # Dataset related configs
    dataset_artifact_path: str = "dataset_info"
