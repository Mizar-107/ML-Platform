"""Checkpoint management callback for HuggingFace Trainer.

Provides S3-based checkpoint management with resume support.
"""

import os
import shutil
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from src.common.logging import get_logger

logger = get_logger(__name__)


class CheckpointCallback(TrainerCallback):
    """Callback for advanced checkpoint management.

    Handles checkpoint saving to S3, cleanup, and resume functionality.

    Attributes:
        checkpoint_dir: Local directory for checkpoints
        s3_bucket: S3 bucket for checkpoint storage
        s3_prefix: S3 prefix for checkpoints
        keep_n_checkpoints: Number of best checkpoints to keep
        metric_for_best: Metric to use for checkpoint ranking
        greater_is_better: Whether higher metric is better
        save_on_each_node: Save checkpoints on each node (DDP)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "./checkpoints",
        s3_bucket: str | None = None,
        s3_prefix: str = "checkpoints",
        keep_n_checkpoints: int = 3,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
        save_on_each_node: bool = False,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Local directory for checkpoints
            s3_bucket: S3 bucket for checkpoint storage
            s3_prefix: S3 prefix for checkpoints
            keep_n_checkpoints: Number of best checkpoints to keep
            metric_for_best: Metric to use for checkpoint ranking
            greater_is_better: Whether higher metric is better
            save_on_each_node: Save checkpoints on each node (DDP)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.keep_n_checkpoints = keep_n_checkpoints
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.save_on_each_node = save_on_each_node

        self._checkpoints: list[dict[str, Any]] = []
        self._s3_client = None

    def _get_s3_client(self):
        """Get or create S3 client."""
        if self._s3_client is None and self.s3_bucket:
            try:
                import boto3

                self._s3_client = boto3.client("s3")
            except ImportError:
                logger.warning("boto3 not installed, S3 upload disabled")
        return self._s3_client

    def _upload_to_s3(self, local_path: Path, s3_key: str) -> bool:
        """Upload a file or directory to S3.

        Args:
            local_path: Local file or directory path
            s3_key: S3 key prefix

        Returns:
            True if upload successful
        """
        client = self._get_s3_client()
        if not client:
            return False

        try:
            if local_path.is_file():
                client.upload_file(str(local_path), self.s3_bucket, s3_key)
            else:
                # Upload directory recursively
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        file_key = f"{s3_key}/{relative_path}"
                        client.upload_file(str(file_path), self.s3_bucket, file_key)

            logger.debug(
                "Uploaded to S3",
                local_path=str(local_path),
                s3_key=s3_key,
            )
            return True

        except Exception as e:
            logger.error(
                "S3 upload failed",
                local_path=str(local_path),
                error=str(e),
            )
            return False

    def _download_from_s3(self, s3_key: str, local_path: Path) -> bool:
        """Download a file or directory from S3.

        Args:
            s3_key: S3 key prefix
            local_path: Local destination path

        Returns:
            True if download successful
        """
        client = self._get_s3_client()
        if not client:
            return False

        try:
            # List objects with prefix
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_key)

            for page in pages:
                for obj in page.get("Contents", []):
                    obj_key = obj["Key"]
                    relative_path = obj_key[len(s3_key) :].lstrip("/")

                    if relative_path:
                        file_path = local_path / relative_path
                    else:
                        file_path = local_path

                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(self.s3_bucket, obj_key, str(file_path))

            logger.debug(
                "Downloaded from S3",
                s3_key=s3_key,
                local_path=str(local_path),
            )
            return True

        except Exception as e:
            logger.error(
                "S3 download failed",
                s3_key=s3_key,
                error=str(e),
            )
            return False

    def _delete_from_s3(self, s3_key: str) -> bool:
        """Delete objects from S3.

        Args:
            s3_key: S3 key prefix

        Returns:
            True if deletion successful
        """
        client = self._get_s3_client()
        if not client:
            return False

        try:
            # List and delete objects with prefix
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_key)

            objects_to_delete = []
            for page in pages:
                for obj in page.get("Contents", []):
                    objects_to_delete.append({"Key": obj["Key"]})

            if objects_to_delete:
                client.delete_objects(
                    Bucket=self.s3_bucket,
                    Delete={"Objects": objects_to_delete},
                )

            logger.debug("Deleted from S3", s3_key=s3_key)
            return True

        except Exception as e:
            logger.error(
                "S3 deletion failed",
                s3_key=s3_key,
                error=str(e),
            )
            return False

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the best N."""
        if len(self._checkpoints) <= self.keep_n_checkpoints:
            return

        # Sort by metric
        sorted_checkpoints = sorted(
            self._checkpoints,
            key=lambda x: x.get("metric", float("-inf" if self.greater_is_better else "inf")),
            reverse=self.greater_is_better,
        )

        # Keep best N, remove rest
        to_remove = sorted_checkpoints[self.keep_n_checkpoints :]
        self._checkpoints = sorted_checkpoints[: self.keep_n_checkpoints]

        for checkpoint in to_remove:
            local_path = Path(checkpoint["path"])

            # Remove local checkpoint
            if local_path.exists():
                shutil.rmtree(local_path)
                logger.debug(
                    "Removed local checkpoint",
                    path=str(local_path),
                )

            # Remove from S3
            if self.s3_bucket and "s3_key" in checkpoint:
                self._delete_from_s3(checkpoint["s3_key"])

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Called when saving a checkpoint.

        Uploads checkpoint to S3 and manages cleanup.
        """
        # Skip if not main process (unless save_on_each_node)
        if not self.save_on_each_node:
            if hasattr(args, "local_rank") and args.local_rank not in [-1, 0]:
                return

        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        if not checkpoint_path.exists():
            return

        # Get metric value for ranking
        metric_value = None
        if state.log_history:
            for log in reversed(state.log_history):
                if self.metric_for_best in log:
                    metric_value = log[self.metric_for_best]
                    break

        checkpoint_info = {
            "step": state.global_step,
            "path": str(checkpoint_path),
            "metric": metric_value,
        }

        # Upload to S3
        if self.s3_bucket:
            s3_key = f"{self.s3_prefix}/checkpoint-{state.global_step}"
            if self._upload_to_s3(checkpoint_path, s3_key):
                checkpoint_info["s3_key"] = s3_key

        self._checkpoints.append(checkpoint_info)
        self._cleanup_old_checkpoints()

        logger.info(
            "Checkpoint saved",
            step=state.global_step,
            metric=metric_value,
            path=str(checkpoint_path),
        )

    def get_latest_checkpoint(self) -> Path | None:
        """Get the path to the latest checkpoint.

        Returns:
            Path to latest checkpoint or None
        """
        if not self._checkpoints:
            # Check local directory
            if self.checkpoint_dir.exists():
                checkpoints = sorted(
                    self.checkpoint_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]),
                )
                if checkpoints:
                    return checkpoints[-1]
            return None

        latest = max(self._checkpoints, key=lambda x: x["step"])
        return Path(latest["path"])

    def get_best_checkpoint(self) -> Path | None:
        """Get the path to the best checkpoint by metric.

        Returns:
            Path to best checkpoint or None
        """
        if not self._checkpoints:
            return None

        valid_checkpoints = [c for c in self._checkpoints if c.get("metric") is not None]
        if not valid_checkpoints:
            return self.get_latest_checkpoint()

        best = (
            max(valid_checkpoints, key=lambda x: x["metric"])
            if self.greater_is_better
            else min(valid_checkpoints, key=lambda x: x["metric"])
        )
        return Path(best["path"])

    def resume_from_s3(
        self,
        checkpoint_name: str | None = None,
    ) -> Path | None:
        """Download and resume from S3 checkpoint.

        Args:
            checkpoint_name: Specific checkpoint name (e.g., "checkpoint-1000")
                           If None, downloads latest

        Returns:
            Path to downloaded checkpoint or None
        """
        if not self.s3_bucket:
            return None

        client = self._get_s3_client()
        if not client:
            return None

        try:
            if checkpoint_name:
                s3_key = f"{self.s3_prefix}/{checkpoint_name}"
            else:
                # Find latest checkpoint in S3
                paginator = client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=self.s3_bucket,
                    Prefix=f"{self.s3_prefix}/checkpoint-",
                    Delimiter="/",
                )

                checkpoints = []
                for page in pages:
                    for prefix in page.get("CommonPrefixes", []):
                        name = prefix["Prefix"].rstrip("/").split("/")[-1]
                        step = int(name.split("-")[-1])
                        checkpoints.append((step, name))

                if not checkpoints:
                    return None

                latest = max(checkpoints, key=lambda x: x[0])
                s3_key = f"{self.s3_prefix}/{latest[1]}"
                checkpoint_name = latest[1]

            # Download checkpoint
            local_path = self.checkpoint_dir / checkpoint_name
            if self._download_from_s3(s3_key, local_path):
                logger.info(
                    "Resumed checkpoint from S3",
                    s3_key=s3_key,
                    local_path=str(local_path),
                )
                return local_path

        except Exception as e:
            logger.error("Failed to resume from S3", error=str(e))

        return None


class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping based on eval metric.

    Stops training when the monitored metric stops improving.

    Attributes:
        patience: Number of evaluations with no improvement before stopping
        threshold: Minimum change to qualify as an improvement
        metric_for_best: Metric to monitor
        greater_is_better: Whether higher metric is better
    """

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.0,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        """Initialize early stopping callback.

        Args:
            patience: Number of evaluations without improvement
            threshold: Minimum change to qualify as improvement
            metric_for_best: Metric to monitor
            greater_is_better: Whether higher metric is better
        """
        self.patience = patience
        self.threshold = threshold
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better

        self._best_metric: float | None = None
        self._patience_counter = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called after evaluation.

        Checks for improvement and updates early stopping state.
        """
        if metrics is None or self.metric_for_best not in metrics:
            return

        current_metric = metrics[self.metric_for_best]

        if self._best_metric is None:
            self._best_metric = current_metric
            return

        # Check for improvement
        if self.greater_is_better:
            improved = current_metric > self._best_metric + self.threshold
        else:
            improved = current_metric < self._best_metric - self.threshold

        if improved:
            self._best_metric = current_metric
            self._patience_counter = 0
            logger.debug(
                "Metric improved",
                metric=self.metric_for_best,
                value=current_metric,
            )
        else:
            self._patience_counter += 1
            logger.debug(
                "No improvement",
                metric=self.metric_for_best,
                patience=f"{self._patience_counter}/{self.patience}",
            )

            if self._patience_counter >= self.patience:
                control.should_training_stop = True
                logger.info(
                    "Early stopping triggered",
                    best_metric=self._best_metric,
                    patience=self.patience,
                )
