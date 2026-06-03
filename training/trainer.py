from typing import Any, Dict, List, Optional

from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig, SFTTrainer
import mlflow
import torch

from .config import ModelConfig, TrainingConfig


class MLflowStepCallback(TrainerCallback):
    """
    Streams per-step training metrics to the active MLflow run in real time.

    Logs on every trainer log event (controlled by TrainingConfig.logging_steps):
      - step_train_loss
      - step_grad_norm
      - step_learning_rate
      - step_epoch
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not logs:
            return
        step = state.global_step
        metrics: Dict[str, float] = {}
        if "loss" in logs:
            metrics["step_train_loss"] = logs["loss"]
        if "grad_norm" in logs:
            metrics["step_grad_norm"] = logs["grad_norm"]
        if "learning_rate" in logs:
            metrics["step_learning_rate"] = logs["learning_rate"]
        if "epoch" in logs:
            metrics["step_epoch"] = logs["epoch"]
        if metrics:
            mlflow.log_metrics(metrics, step=step)


class LoRATrainer:
    """Wraps HuggingFace SFTTrainer with project-specific configuration."""

    def __init__(
        self,
        model,
        tokenizer,
        training_config: TrainingConfig,
        model_config: ModelConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.model_config = model_config
        self._trainer: Optional[SFTTrainer] = None
        self._train_result = None

    def _build_sft_config(self) -> SFTConfig:
        cfg = self.training_config
        return SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            max_length=self.model_config.max_seq_length,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            seed=cfg.seed,
            report_to="none",
            dataset_text_field="text",
            packing=False,
        )

    def train(self, train_dataset: Dataset) -> Dict[str, Any]:
        """
        Fit the LoRA model on train_dataset.

        Each logged step is streamed to MLflow in real-time via MLflowStepCallback.
        Returns the raw metrics dict from TrainOutput.
        """
        sft_config = self._build_sft_config()
        self._trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            args=sft_config,
            processing_class=self.tokenizer,
            callbacks=[MLflowStepCallback()],
        )
        cfg = self.training_config
        print("Starting LoRA fine-tuning …")
        print(f"  Epochs:         {cfg.num_train_epochs}")
        print(f"  Effective batch:{cfg.effective_batch_size}  "
              f"({cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps})")
        print(f"  Learning rate:  {cfg.learning_rate}")

        self._train_result = self._trainer.train()
        metrics = self._train_result.metrics
        print(f"\nTraining complete.")
        print(f"  Steps:        {self._train_result.global_step}")
        print(f"  Runtime:      {metrics['train_runtime']:.0f}s")
        print(f"  Final loss:   {metrics['train_loss']:.4f}")
        return metrics

    def get_log_history(self) -> List[Dict[str, Any]]:
        """Return the per-step log entries recorded by the Trainer callback."""
        if self._trainer is None:
            return []
        return self._trainer.state.log_history

    def get_step_losses(self) -> List[Dict[str, Any]]:
        """Filter log history to only entries that contain a training loss value."""
        return [e for e in self.get_log_history() if "loss" in e]
