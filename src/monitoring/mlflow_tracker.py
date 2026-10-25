"""
MLflow Experiment Tracking and Model Registry Management.
Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine with Closed-Loop ERP Action.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLOpsConfig, DEFAULT_MLOPS_CONFIG
from src.models.engine import PrognosticsEngine
from src.models.isolation_forest import IsolationForestBaseline

# Silence MLflow interactive agent hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


class MLflowTracker:
    """
    MLflow Tracking and Model Registry Controller.
    Manages parameters, metrics, model artifacts, and version promotion for the prognostics engine.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[MLOpsConfig] = None,
    ):
        self.config = config or DEFAULT_MLOPS_CONFIG
        self.tracking_uri = tracking_uri or self.config.tracking_uri
        self.experiment_name = experiment_name or self.config.experiment_name

        self._setup_tracking_environment()
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_id = self._get_or_create_experiment()

    def _setup_tracking_environment(self) -> None:
        """Configure MLflow backend store and artifact repository paths."""
        if self.tracking_uri.startswith("sqlite:///"):
            raw_path = self.tracking_uri.replace("sqlite:///", "")
            # Ensure parent directory exists for SQLite database file
            db_dir = os.path.dirname(raw_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            # Standardize absolute path with forward slashes for Windows compatibility
            abs_db = os.path.abspath(raw_path).replace("\\", "/")
            self.tracking_uri = f"sqlite:///{abs_db}"

        mlflow.set_tracking_uri(self.tracking_uri)

    def _get_or_create_experiment(self) -> str:
        """Ensure experiment exists and return its unique ID."""
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id

        return self.client.create_experiment(name=self.experiment_name)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> mlflow.ActiveRun:
        """Start a new tracked MLflow run."""
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
            nested=nested,
        )

    def end_run(self, status: str = "FINISHED") -> None:
        """End the currently active MLflow run."""
        if mlflow.active_run():
            mlflow.end_run(status=status)

    @contextmanager
    def track_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """Context manager for automatically starting and completing an MLflow run."""
        run = self.start_run(run_name=run_name, tags=tags, nested=nested)
        try:
            yield run
        except Exception as e:
            self.end_run(status="FAILED")
            raise e
        else:
            self.end_run(status="FINISHED")

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter or configuration attribute."""
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a numerical evaluation metric."""
        mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of evaluation metrics."""
        clean_metrics = {k: float(v) for k, v in metrics.items() if not np.isnan(v)}
        mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Upload a local file or directory as a run artifact."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a JSON-serializable dictionary directly to run artifacts."""
        mlflow.log_dict(dictionary, artifact_file)

    def log_isolation_forest(
        self,
        model: IsolationForestBaseline,
        artifact_path: str = "isolation_forest",
    ) -> None:
        """Serialize and log Isolation Forest baseline detector."""
        temp_dir = tempfile.mkdtemp(prefix="iso_forest_")
        try:
            model_file = os.path.join(temp_dir, "iso_forest.joblib")
            model.save(model_file)
            self.log_artifact(model_file, artifact_path=artifact_path)
            self.log_params({
                "iso_forest_contamination": model.contamination,
                "iso_forest_n_estimators": model.n_estimators,
                "iso_forest_random_state": model.random_state,
                "iso_forest_num_features": len(model.feature_cols),
            })
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def log_prognostics_engine(
        self,
        engine: PrognosticsEngine,
        artifact_path: str = "prognostics_engine",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Serialize and log complete 4-model prognostics engine bundle as MLflow artifacts.
        Returns the logged artifact relative path.
        """
        temp_dir = tempfile.mkdtemp(prefix="prognostics_engine_")
        try:
            engine.save(temp_dir)
            if metadata:
                import json
                meta_file = os.path.join(temp_dir, "engine_metadata.json")
                with open(meta_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            self.log_artifact(temp_dir, artifact_path=artifact_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return artifact_path

    def register_model(
        self,
        run_id: str,
        model_name: Optional[str] = None,
        artifact_path: str = "prognostics_engine",
    ) -> Any:
        """
        Register a logged run model into the MLflow Model Registry.
        """
        target_name = model_name or self.config.model_registry_name
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.register_model(model_uri=model_uri, name=target_name)

    def get_latest_model_versions(
        self,
        model_name: Optional[str] = None,
        stages: Optional[List[str]] = None,
    ) -> List[Any]:
        """Retrieve latest versions of registered model for specified stages."""
        target_name = model_name or self.config.model_registry_name
        try:
            return self.client.get_latest_versions(name=target_name, stages=stages)
        except Exception:
            return []

    def transition_model_stage(
        self,
        version: Union[int, str],
        stage: str,
        model_name: Optional[str] = None,
        archive_existing_versions: bool = True,
    ) -> Any:
        """
        Promote or transition a registered model version to 'Staging', 'Production', or 'Archived'.
        """
        target_name = model_name or self.config.model_registry_name
        return self.client.transition_model_version_stage(
            name=target_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    def get_best_run(
        self,
        metric_name: str = "test_f1_macro",
        ascending: bool = False,
    ) -> Optional[mlflow.entities.Run]:
        """Search and retrieve best performing MLflow run based on specified metric."""
        order_direction = "ASC" if ascending else "DESC"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {order_direction}"],
            max_results=1,
        )
        return runs[0] if runs else None

    def list_runs(self, max_results: int = 10) -> List[mlflow.entities.Run]:
        """List recent runs for this experiment."""
        return self.client.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=max_results,
        )
