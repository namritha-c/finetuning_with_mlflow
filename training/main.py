"""
Entry point for LoRA fine-tuning with MLflow experiment tracking.

Usage:
    python -m training.main          (recommended, from project root)
    python training/main.py          (also works, from project root)
"""

import os
import random
import sys
import time

import torch

# Ensure the project root (parent of this package) is on sys.path so that
# `from training import ...` resolves correctly when the script is run
# directly as `python training/main.py` instead of `python -m training.main`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training import (
    DataConfig,
    DatasetLoader,
    LoRAConfig,
    LoRATrainer,
    MLflowConfig,
    MLflowLogger,
    ModelConfig,
    ModelEvaluator,
    ModelLoader,
    PromptConfig,
    TrainingConfig,
    Visualizer,
)


def _setup_environment(seed: int = 42) -> None:
    # CUDA_VISIBLE_DEVICES and TOKENIZERS_PARALLELISM are already in os.environ
    # via load_dotenv() called at config import time; setdefault is a safe fallback (Use if needed).

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

    # All local output files (plots, etc.) go here — keeps the project root clean.
    plots_dir = os.path.join(_PROJECT_ROOT, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. MLflow — open training run and log all configs upfront
    # ------------------------------------------------------------------
    mlflow_logger = MLflowLogger(mlflow_config)
    mlflow_logger.start_run()

    # Logging using log_params() method
    mlflow_logger.log_model_config(model_config)
    mlflow_logger.log_lora_config(lora_config)
    mlflow_logger.log_training_config(training_config)
    mlflow_logger.log_data_config(data_config)

    # ------------------------------------------------------------------
    # 3. System prompt — register in MLflow Prompt Registry, then load back
    # ------------------------------------------------------------------
    print("\n--- Registering system prompt ---")
    mlflow_logger.register_system_prompt(prompt_config)
    system_prompt = mlflow_logger.load_system_prompt_from_registry(prompt_config)

    # ------------------------------------------------------------------
    # 4. Dataset — load and log to MLflow with full lineage
    # ------------------------------------------------------------------
    print("\n--- Loading dataset ---")
    data_loader = DatasetLoader(data_config, system_prompt=system_prompt)
    data_loader.load()

    mlflow_logger.log_dataset(data_loader.train_data, data_loader.test_data, data_config)

    # ------------------------------------------------------------------
    # 5. Base model — load and evaluate BEFORE fine-tuning
    # ------------------------------------------------------------------
    print("\n--- Loading base model ---")
    model_loader = ModelLoader(model_config, lora_config)
    model, tokenizer = model_loader.load_base_model()

    print("\n--- Evaluating base model ---")
    base_evaluator = ModelEvaluator(model, tokenizer, data_loader)
    base_eval = base_evaluator.evaluate(label="Base Model")
    base_evaluator.print_sample_outputs(base_eval, n=3, label="Base Model")

    # ------------------------------------------------------------------
    # 6. Apply LoRA and prepare training data
    # ------------------------------------------------------------------
    print("\n--- Applying LoRA adapters ---")
    model_loader.apply_lora(seed=training_config.seed)
    total_params, trainable_params = model_loader.get_parameter_counts()
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.4f}%)")


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

    # ------------------------------------------------------------------
    # 9. Log final model to MLflow (transformers flavor)
    # ------------------------------------------------------------------
    print("\n--- Logging final model to MLflow ---")
    mlflow_logger.log_final_model(model_loader, tokenizer)

    # Capture the training run ID before closing the run so the eval run
    # can reference it in its tags.
    training_run_id = mlflow_logger.run_id
    mlflow_logger.end_run()

    # ------------------------------------------------------------------
    # 10. Evaluation run — separate run tagged as "evaluation"
    # ------------------------------------------------------------------
    mlflow_logger.start_eval_run(training_run_id)

    print("\n--- Evaluating fine-tuned model ---")
    ft_evaluator = ModelEvaluator(model_loader.model, tokenizer, data_loader)
    finetuned_eval = ft_evaluator.evaluate(label="Fine-Tuned Model")
    ft_evaluator.print_sample_outputs(finetuned_eval, n=3, label="Fine-Tuned Model")

    # ------------------------------------------------------------------
    # 11. Log evaluation metrics and visualisations
    # ------------------------------------------------------------------
    mlflow_logger.log_evaluation_metrics(
        base_eval, finetuned_eval, total_params, trainable_params
    )

    acc_plot_path = os.path.join(plots_dir, "accuracy_comparison.png")
    Visualizer.plot_accuracy_comparison(
        base_eval["accuracy"], finetuned_eval["accuracy"], save_path=acc_plot_path
    )
    mlflow_logger.log_artifact(acc_plot_path, artifact_subdir="plots")

    summary_plot_path = os.path.join(plots_dir, "summary.png")
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
    # 12. Print final summary and close eval run
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
