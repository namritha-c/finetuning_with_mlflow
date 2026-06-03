"""
Entry point for LoRA fine-tuning with MLflow experiment tracking.

Usage:
    python -m training.main
    # or from Fine_tuning/
    python training/main.py
"""

import os
import random
import time

import torch

from training.config import DataConfig, LoRAConfig, MLflowConfig, ModelConfig, PromptConfig, TrainingConfig
from training.data_loader import DatasetLoader
from training.evaluator import ModelEvaluator
from training.mlflow_logger import MLflowLogger
from training.model_loader import ModelLoader
from training.trainer import LoRATrainer
from training.visualizer import Visualizer


def _setup_environment(seed: int = 42) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Configs
    # ------------------------------------------------------------------
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()
    mlflow_config = MLflowConfig()
    prompt_config = PromptConfig()

    _setup_environment(seed=data_config.seed)

    # ------------------------------------------------------------------
    # 2. MLflow — open run and log all configs upfront
    # ------------------------------------------------------------------
    mlflow_logger = MLflowLogger(mlflow_config)
    mlflow_logger.start_run(run_name=f"lora_sql_{int(time.time())}")

    mlflow_logger.log_model_config(model_config)
    mlflow_logger.log_lora_config(lora_config)
    mlflow_logger.log_training_config(training_config)
    mlflow_logger.log_data_config(data_config)

    # ------------------------------------------------------------------
    # 3. System prompt — load from a previous MLflow run or use the default
    # ------------------------------------------------------------------
    if mlflow_config.load_prompt_from_run_id:
        system_prompt = mlflow_logger.load_system_prompt(
            mlflow_config.load_prompt_from_run_id, prompt_config
        )
    else:
        system_prompt = prompt_config.text
        mlflow_logger.log_system_prompt(prompt_config)

    # ------------------------------------------------------------------
    # 4. Dataset
    # ------------------------------------------------------------------
    print("\n--- Loading dataset ---")
    data_loader = DatasetLoader(data_config, system_prompt=system_prompt)
    data_loader.load()

    mlflow_logger.log_dataset_artifact(data_loader.train_data, data_loader.test_data)

    # ------------------------------------------------------------------
    # 5. Base model — load and evaluate BEFORE fine-tuning
    # ------------------------------------------------------------------
    print("\n--- Loading base model ---")
    model_loader = ModelLoader(model_config, lora_config)
    model, tokenizer = model_loader.load_base_model()

    print("\n--- Evaluating base model ---")
    base_evaluator = ModelEvaluator(model, tokenizer, data_loader)
    base_eval = base_evaluator.evaluate(data_loader.test_data, label="Base Model")
    base_evaluator.print_sample_outputs(base_eval, n=3, label="Base Model")

    # ------------------------------------------------------------------
    # 6. Apply LoRA and prepare training data
    # ------------------------------------------------------------------
    print("\n--- Applying LoRA adapters ---")
    model_loader.apply_lora(seed=training_config.seed)
    total_params, trainable_params = model_loader.get_parameter_counts()

    print("\n--- Preparing training dataset ---")
    train_dataset = data_loader.prepare_training_dataset(tokenizer)
    print(f"Training dataset ready: {len(train_dataset):,} examples")

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("\n--- Training ---")
    lora_trainer = LoRATrainer(model_loader.model, tokenizer, training_config, model_config)
    train_metrics = lora_trainer.train(train_dataset)

    mlflow_logger.log_training_metrics(train_metrics)
    mlflow_logger.log_step_losses(lora_trainer.get_log_history())

    # ------------------------------------------------------------------
    # 8. Plots — training loss
    # ------------------------------------------------------------------
    loss_plot_path = "training_loss.png"
    Visualizer.plot_training_loss(lora_trainer.get_log_history(), save_path=loss_plot_path)
    mlflow_logger.log_artifact(loss_plot_path, artifact_subdir="plots")

    # ------------------------------------------------------------------
    # 9. Fine-tuned model evaluation
    # ------------------------------------------------------------------
    print("\n--- Evaluating fine-tuned model ---")
    ft_evaluator = ModelEvaluator(model_loader.model, tokenizer, data_loader)
    finetuned_eval = ft_evaluator.evaluate(data_loader.test_data, label="Fine-Tuned Model")
    ft_evaluator.print_sample_outputs(finetuned_eval, n=3, label="Fine-Tuned Model")

    # ------------------------------------------------------------------
    # 10. Log evaluation metrics and visualisations
    # ------------------------------------------------------------------
    mlflow_logger.log_evaluation_metrics(
        base_eval, finetuned_eval, total_params, trainable_params
    )

    acc_plot_path = "accuracy_comparison.png"
    Visualizer.plot_accuracy_comparison(
        base_eval["accuracy"], finetuned_eval["accuracy"], save_path=acc_plot_path
    )
    mlflow_logger.log_artifact(acc_plot_path, artifact_subdir="plots")

    summary_plot_path = "summary.png"
    Visualizer.plot_summary(
        base_acc=base_eval["accuracy"],
        ft_acc=finetuned_eval["accuracy"],
        total_params=total_params,
        trainable_params=trainable_params,
        train_runtime_seconds=train_metrics.get("train_runtime", 0.0),
        save_path=summary_plot_path,
    )
    mlflow_logger.log_artifact(summary_plot_path, artifact_subdir="plots")

    # ------------------------------------------------------------------
    # 11. Log LoRA adapter weights to MLflow
    # ------------------------------------------------------------------
    print("\n--- Logging model adapter to MLflow ---")
    mlflow_logger.log_model_adapter(model_loader)

    # ------------------------------------------------------------------
    # 12. Print final summary and close run
    # ------------------------------------------------------------------
    base_acc = base_eval["accuracy"]
    ft_acc = finetuned_eval["accuracy"]
    print("\n" + "=" * 55)
    print(f"{'METRIC':<28} {'BASE':>10} {'FINE-TUNED':>12}")
    print("=" * 55)
    print(f"{'Exact-Match Accuracy':<28} {base_acc:>9.1f}% {ft_acc:>11.1f}%")
    print(f"{'Correct / Total':<28} {base_eval['correct']:>4}/{base_eval['total']:<5} "
          f"{finetuned_eval['correct']:>4}/{finetuned_eval['total']:<5}")
    print("=" * 55)
    print(f"\nAccuracy improvement : +{ft_acc - base_acc:.1f} pp")
    print(f"Parameters trained   : {trainable_params:,} / {total_params:,} "
          f"({trainable_params / total_params * 100:.2f}%)")
    print(f"Training time        : {train_metrics.get('train_runtime', 0):.0f}s")

    mlflow_logger.end_run()


if __name__ == "__main__":
    main()
