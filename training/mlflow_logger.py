import json
import os
import shutil
import time
import traceback
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.artifacts
import mlflow.genai
import mlflow.pyfunc


from .config import DataConfig, LoRAConfig, MLflowConfig, ModelConfig, PromptConfig, TrainingConfig
from .model_loader import ModelLoader


class _LoRAAdapterModel(mlflow.pyfunc.PythonModel):
    """
    Minimal MLflow PythonModel wrapper for a LoRA adapter.

    Packages the adapter weights as a first-class MLflow model so the run
    artifact has a valid MLmodel file and can be registered and versioned in
    the MLflow Model Registry.

    To load the adapter for inference:
        import mlflow.pyfunc
        from peft import PeftModel
        loaded = mlflow.pyfunc.load_model("models:/lora-sql-qwen2.5/latest")
        adapter_path = loaded._model_impl.context.artifacts["lora_adapter"]
        model = PeftModel.from_pretrained(base_model, adapter_path)
    """

    def predict(self, context, model_input, params=None):
        return {"adapter_path": context.artifacts["lora_adapter"]}


class MLflowLogger:
    """
    Centralises all MLflow interactions: experiment setup, param/metric logging,
    artifact uploads, dataset tracking, model logging, and eval run management.
    """

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config
        self._active_run: Optional[mlflow.ActiveRun] = None
        self._configure_auth()
        mlflow.set_tracking_uri(config.tracking_uri)
        self._setup_experiment()

    def _configure_auth(self) -> None:
        """
        Push MLflow credentials into os.environ so the HTTP tracking client
        includes them in every request. This is a no-op when the variables are
        already present (e.g. set directly in the shell), so it is safe to call
        unconditionally.
        """
        if self.config.username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.username
        if self.config.password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.password
        if self.config.username:
            print(f"MLflow auth configured for user: '{self.config.username}'")

    # ------------------------------------------------------------------
    # Experiment / run lifecycle
    # ------------------------------------------------------------------

    def _setup_experiment(self) -> None:
        """Create the experiment if it does not already exist, then set it as active."""
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.config.experiment_name,
                tags={"project": self.config.project_name, "framework": "unsloth+trl"},
                description=self.config.description,
            )
            print(f"Created MLflow experiment: '{self.config.experiment_name}'")
        else:
            print(f"Using existing MLflow experiment: '{self.config.experiment_name}'")
        mlflow.set_experiment(self.config.experiment_name)

    @property
    def run_id(self) -> Optional[str]:
        """Return the active run ID, or None if no run is open."""
        return self._active_run.info.run_id if self._active_run else None

    def start_run(self) -> None:
        """Begin a new training run under the configured experiment."""
        run_name=f"{self.config.run_name}_{int(time.time())}"
        self._active_run = mlflow.start_run(
            run_name=run_name,
            log_system_metrics=True,
            tags={"run_type": "training"},
        )
        print(f"MLflow run started — ID: {self._active_run.info.run_id}")

    def start_eval_run(self, training_run_id: str, run_name: Optional[str] = None) -> None:
        """
        Open a separate evaluation run in the same experiment.

        The run is tagged with run_type=evaluation and linked back to the
        training run via the training_run_id tag so both are easily correlated
        in the MLflow UI.
        """
        name = run_name or f"eval_{training_run_id[:8]}"
        self._active_run = mlflow.start_run(
            run_name=name,
            tags={
                "run_type": "evaluation",
                "training_run_id": training_run_id,
            },
        )
        print(f"MLflow eval run started — ID: {self._active_run.info.run_id}")

    def end_run(self) -> None:
        """Finalise the current run and print a link to the UI."""
        if self._active_run:
            run_id = self._active_run.info.run_id
            mlflow.end_run()
            exp = mlflow.get_experiment_by_name(self.config.experiment_name)
            ui_url = (
                f"{self.config.tracking_uri}/#/experiments/"
                f"{exp.experiment_id}/runs/{run_id}"
            )
            print(f"MLflow run complete.\n  View at: {ui_url}")
            self._active_run = None

    # ------------------------------------------------------------------
    # Config / parameter logging
    # ------------------------------------------------------------------

    def log_model_config(self, config: ModelConfig) -> None:
        mlflow.log_params({
            "model_dtype": str(config.dtype),
            "model_load_in_4bit": config.load_in_4bit,
            "model_max_seq_length": config.max_seq_length,
            "model_name": config.name,
        })

    def log_lora_config(self, config: LoRAConfig) -> None:
        mlflow.log_params({
            "lora_alpha": config.lora_alpha,
            "lora_bias": config.bias,
            "lora_dropout": config.lora_dropout,
            "lora_r": config.r,
            "lora_target_modules": ",".join(config.target_modules),
            "lora_use_gradient_checkpointing": config.use_gradient_checkpointing,
        })

    def log_training_config(self, config: TrainingConfig) -> None:
        mlflow.log_params({
            "training_effective_batch_size": config.effective_batch_size,
            "training_gradient_accumulation_steps": config.gradient_accumulation_steps,
            "training_learning_rate": config.learning_rate,
            "training_lr_scheduler_type": config.lr_scheduler_type,
            "training_num_epochs": config.num_train_epochs,
            "training_per_device_batch_size": config.per_device_train_batch_size,
            "training_save_strategy": config.save_strategy,
            "training_seed": config.seed,
            "training_warmup_ratio": config.warmup_ratio,
        })

    def log_data_config(self, config: DataConfig) -> None:
        mlflow.log_params({
            "data_dataset_name": config.dataset_name,
            "data_num_test": config.num_test,
            "data_num_train": config.num_train,
            "data_seed": config.seed,
        })

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log scalar summaries from trainer.train() metrics dict."""
        mlflow.log_metrics({
            "train_loss_final": metrics.get("train_loss", 0.0),
            "train_runtime_seconds": metrics.get("train_runtime", 0.0),
            "train_samples_per_second": metrics.get("train_samples_per_second", 0.0),
            "train_steps_per_second": metrics.get("train_steps_per_second", 0.0),
        })

    def log_evaluation_metrics(
        self,
        base_eval: Dict[str, Any],
        finetuned_eval: Dict[str, Any],
        total_params: int,
        trainable_params: int,
    ) -> None:
        mlflow.log_metrics({
            "eval_base_accuracy": base_eval["accuracy"],
            "eval_base_correct": base_eval["correct"],
            "eval_base_elapsed_seconds": base_eval["elapsed_seconds"],
            "eval_finetuned_accuracy": finetuned_eval["accuracy"],
            "eval_finetuned_correct": finetuned_eval["correct"],
            "eval_finetuned_elapsed_seconds": finetuned_eval["elapsed_seconds"],
            "eval_accuracy_improvement": finetuned_eval["accuracy"] - base_eval["accuracy"],
            "model_total_parameters": float(total_params),
            "model_trainable_parameters": float(trainable_params),
            "model_trainable_percentage": trainable_params / total_params * 100,
        })

    # ------------------------------------------------------------------
    # Artifact logging
    # ------------------------------------------------------------------

    def log_artifact(self, local_path: str, artifact_subdir: str = "") -> None:
        """Upload a local file to MLflow artifacts (silently skips missing files)."""
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path=artifact_subdir or None)
        else:
            print(f"Warning: artifact not found, skipping: {local_path}")

    # ------------------------------------------------------------------
    # Prompt registry
    # ------------------------------------------------------------------

    def register_system_prompt(self, prompt_config: PromptConfig) -> None:
        """
        Register (or create a new version of) the system prompt in the
        MLflow Prompt Registry under the name set by MLflowConfig.prompt_registry_name.

        Versions are immutable once created — MLflow auto-increments the version
        number each time this is called with a changed template.
        """
        prompt = mlflow.genai.register_prompt(
            name=self.config.prompt_registry_name,
            template=prompt_config.text,
            commit_message=prompt_config.commit_message,
            tags={"task": "text-to-sql", "role": "system"},
        )
        print(
            f"System prompt registered in MLflow Prompt Registry: "
            f"'{self.config.prompt_registry_name}' v{prompt.version}"
        )

    def load_system_prompt_from_registry(self, prompt_config: PromptConfig) -> str:
        """
        Load the latest version of the system prompt from the MLflow Prompt Registry
        and return the raw template string.

        URI format: prompts:/<name>@latest
        """
        uri = f"prompts:/{self.config.prompt_registry_name}@{prompt_config.version}"
        prompt = mlflow.genai.load_prompt(uri)
        print(
            f"System prompt loaded from MLflow Prompt Registry: "
            f"'{self.config.prompt_registry_name}@latest' (v{prompt.version})"
        )
        return prompt.template

    # ------------------------------------------------------------------
    # Dataset logging
    # ------------------------------------------------------------------

    def log_dataset(
        self,
        train_data,
        test_data,
        data_config: DataConfig,
    ) -> None:
        """
        Log train and test splits to MLflow using the Dataset Tracking API
        (mlflow.log_input + mlflow.data.from_huggingface).

        This creates proper dataset lineage in the MLflow UI — the Inputs tab
        will show source, split, size, and a direct link to the HuggingFace hub.
        Falls back to a JSON artifact when the HuggingFace data module is
        unavailable.
        """
        try:
            from mlflow.data.huggingface_dataset import from_huggingface

            train_ds = from_huggingface(
                train_data,
                path=data_config.dataset_name,
                targets=data_config.target_column,
                name=f"{data_config.dataset_name}_train",
            )
            mlflow.log_input(train_ds, context="training", tags={"split": "train"})

            test_ds = from_huggingface(
                test_data,
                path=data_config.dataset_name,
                targets=data_config.target_column,
                name=f"{data_config.dataset_name}_test",
            )
            mlflow.log_input(test_ds, context="evaluation", tags={"split": "test"})

            print(
                f"Dataset logged via MLflow Data API: "
                f"{len(train_data):,} train / {len(test_data):,} test  "
                f"({data_config.dataset_name})"
            )

        except Exception as exc:
            print(f"Warning: mlflow.data.from_huggingface failed ({exc}), falling back to JSON artifact.")
            self._log_dataset_json_fallback(train_data, test_data)

    def _log_dataset_json_fallback(
        self,
        train_data,
        test_data,
        sample_n: int = 5,
        local_filename: str = "dataset_info.json",
    ) -> None:
        payload = {
            "dataset_name": train_data.info.dataset_name if hasattr(train_data, "info") else "unknown",
            "train_size": len(train_data),
            "test_size": len(test_data),
            "train_columns": train_data.column_names,
            "train_sample": [
                {
                    "question": train_data[i]["question"],
                    "context": train_data[i]["context"][:300],
                    "answer": train_data[i]["answer"],
                }
                for i in range(min(sample_n, len(train_data)))
            ],
        }
        with open(local_filename, "w") as fh:
            json.dump(payload, fh, indent=2)
        mlflow.log_artifact(local_filename, artifact_path=self.config.dataset_artifact_path)
        os.remove(local_filename)
        print(f"Dataset artifact logged to MLflow ({len(train_data)} train, {len(test_data)} test).")

    # ------------------------------------------------------------------
    # Model logging
    # ------------------------------------------------------------------

    def log_final_model(self, model_loader: ModelLoader, tokenizer, tmp_dir: str = "/tmp/lora_adapter") -> None:
        """
        Save the LoRA adapter weights and register them in the MLflow Model Registry.

        mlflow.transformers.log_model is intentionally skipped: 4-bit quantized
        PEFT models (bitsandbytes) cannot be serialised by that flavor.
        Instead, mlflow.pyfunc.log_model wraps the adapter files with a valid
        MLmodel manifest so the artifact can be registered and versioned in the
        Model Registry in one step via registered_model_name.
        """
        # Always start from a clean directory so stale files from a previous
        # run cannot pollute the artifact.
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            model_loader.save_adapter(tmp_dir)

            saved_files = os.listdir(tmp_dir)
            if not saved_files:
                raise RuntimeError(f"save_adapter() produced no files in {tmp_dir}.")
            print(f"Adapter files saved: {saved_files}")

            mlflow.pyfunc.log_model(
                artifact_path=self.config.model_artifact_path,
                python_model=_LoRAAdapterModel(),
                artifacts={"lora_adapter": tmp_dir},
                registered_model_name=self.config.registered_model_name,
                metadata={
                    "base_model": model_loader.model_config.name,
                    "adapter_type": "lora",
                    "task": "text-to-sql",
                    "framework": "unsloth+peft",
                },
            )
            print(
                f"Model logged and registered in MLflow Model Registry: "
                f"'{self.config.registered_model_name}'."
            )
        except Exception as exc:
            print(f"ERROR: could not log/register model in MLflow — {exc}")
            traceback.print_exc()

    def log_step_losses(self, log_history: List[Dict[str, Any]]) -> None:
        """Log per-step training loss as a metric time series (batch, post-training)."""
        for entry in log_history:
            if "loss" in entry:
                mlflow.log_metric("step_train_loss", entry["loss"], step=entry["step"])
