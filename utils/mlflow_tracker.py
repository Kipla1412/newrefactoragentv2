import mlflow

from utils.logging_config import get_logger

logger = get_logger(__name__)

_mlflow_instance = None


def get_mlflow_tracker(config):

    global _mlflow_instance

    if not config.mlflow_enabled:
        logger.info("MLflow tracking disabled")
        return None

    if _mlflow_instance is None:

        try:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)

            logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

            mlflow.set_experiment(
                config.mlflow_experiment_name or "clinical-agent"
            )
    
            try:
                mlflow.openai.autolog()
                logger.info("MLflow autolog enabled")
            except Exception as e:
                logger.warning("MLflow autolog failed: %s", e)

            _mlflow_instance = mlflow

        except Exception as e:

            logger.error("MLflow initialization failed: %s", e)

            return None

    return _mlflow_instance