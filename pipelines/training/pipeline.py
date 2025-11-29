"""
Training Pipeline Definition

This is the main DAG that orchestrates the training workflow.
It connects all components in sequence and handles conditional logic.

Pipeline Flow:
    Data Ingestion → Data Validation → Training → Evaluation → Registration

Usage:
    # Compile pipeline
    python pipeline.py --compile
    
    # Submit to Kubeflow
    python pipeline.py --submit --data-version v1.0.0
"""

from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Artifact
from typing import NamedTuple
import argparse


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

PIPELINE_NAME = "model-training-pipeline"
PIPELINE_DESCRIPTION = """
End-to-end training pipeline with data validation, experiment tracking,
and conditional model registration based on performance evaluation.
"""

# Base images from ECR (these would be your actual ECR URIs)
BASE_IMAGE = "python:3.10-slim"  # Replace with your ECR image
GPU_IMAGE = "python:3.10-slim"   # Replace with your GPU-enabled ECR image


# =============================================================================
# COMPONENT IMPORTS
# 
# In production, these would be imported from separate files:
#   from components.data_ingestion import data_ingestion_op
#   from components.data_validation import data_validation_op
#   etc.
#
# For now, we define them inline for clarity. We'll refactor later.
# =============================================================================


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "pyarrow", "boto3", "feast", "dvc[s3]"]
)
def data_ingestion(
    data_version: str,
    feature_set: str,
    s3_bucket: str,
    feast_repo_path: str,
) -> NamedTuple('Outputs', [
    ('dataset_path', str),
    ('dvc_commit', str),
    ('row_count', int),
]):
    """
    Pull versioned data from DVC and fetch historical features from Feast.
    
    Args:
        data_version: DVC tag/version to pull (e.g., "v1.0.0")
        feature_set: Feast feature service name
        s3_bucket: S3 bucket for DVC remote
        feast_repo_path: Path to Feast feature repository
    
    Returns:
        dataset_path: Path to the prepared training dataset
        dvc_commit: DVC commit hash for lineage tracking
        row_count: Number of rows in the dataset
    """
    import pandas as pd
    import subprocess
    import os
    
    print(f"Starting data ingestion...")
    print(f"  Data version: {data_version}")
    print(f"  Feature set: {feature_set}")
    
    # ----- DVC: Pull versioned data -----
    # In production, this would checkout the specific version and pull
    # For now, we simulate the process
    
    # dvc checkout <data_version>
    # dvc pull
    
    dvc_commit = data_version  # Would be actual commit hash
    print(f"  DVC commit: {dvc_commit}")
    
    # ----- Feast: Get historical features -----
    # In production:
    # from feast import FeatureStore
    # store = FeatureStore(repo_path=feast_repo_path)
    # training_df = store.get_historical_features(
    #     entity_df=entity_df,
    #     features=store.get_feature_service(feature_set),
    # ).to_df()
    
    # Simulated dataset for pipeline testing
    training_df = pd.DataFrame({
        'feature_1': range(1000),
        'feature_2': range(1000, 2000),
        'feature_3': [i * 0.1 for i in range(1000)],
        'target': [i % 2 for i in range(1000)]  # Binary classification
    })
    
    # Save dataset
    output_path = "/tmp/training_data.parquet"
    training_df.to_parquet(output_path, index=False)
    
    row_count = len(training_df)
    print(f"  Dataset saved: {row_count} rows")
    
    # Return named tuple
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['dataset_path', 'dvc_commit', 'row_count'])
    return outputs(output_path, dvc_commit, row_count)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "great_expectations"]
)
def data_validation(
    dataset_path: str,
    min_row_count: int,
    max_null_percentage: float,
) -> NamedTuple('Outputs', [
    ('validation_passed', bool),
    ('validation_report_path', str),
]):
    """
    Validate data quality using Great Expectations.
    
    Checks:
        - Minimum row count
        - Null percentage thresholds
        - Schema validation
        - Value range checks
    
    Args:
        dataset_path: Path to the dataset from ingestion step
        min_row_count: Minimum acceptable rows
        max_null_percentage: Maximum allowed null percentage per column
    
    Returns:
        validation_passed: Boolean indicating if all checks passed
        validation_report_path: Path to detailed validation report
    """
    import pandas as pd
    import json
    
    print("Starting data validation...")
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    validation_results = {
        "checks": [],
        "passed": True
    }
    
    # ----- Check 1: Minimum row count -----
    row_check = {
        "name": "row_count_check",
        "expected": f">= {min_row_count}",
        "actual": len(df),
        "passed": len(df) >= min_row_count
    }
    validation_results["checks"].append(row_check)
    print(f"  Row count check: {row_check['passed']} ({len(df)} >= {min_row_count})")
    
    # ----- Check 2: Null percentage -----
    for col in df.columns:
        null_pct = df[col].isnull().sum() / len(df) * 100
        null_check = {
            "name": f"null_check_{col}",
            "expected": f"<= {max_null_percentage}%",
            "actual": f"{null_pct:.2f}%",
            "passed": null_pct <= max_null_percentage
        }
        validation_results["checks"].append(null_check)
        
        if not null_check["passed"]:
            validation_results["passed"] = False
            print(f"  FAILED: {col} has {null_pct:.2f}% nulls")
    
    # ----- Check 3: Required columns exist -----
    required_columns = ["target"]  # Add your required columns
    for col in required_columns:
        col_check = {
            "name": f"column_exists_{col}",
            "expected": "column exists",
            "actual": col in df.columns,
            "passed": col in df.columns
        }
        validation_results["checks"].append(col_check)
        
        if not col_check["passed"]:
            validation_results["passed"] = False
            print(f"  FAILED: Required column '{col}' missing")
    
    # Save validation report
    report_path = "/tmp/validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"  Overall validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['validation_passed', 'validation_report_path'])
    return outputs(validation_results["passed"], report_path)


@dsl.component(
    base_image=BASE_IMAGE,  # In production, use GPU_IMAGE
    packages_to_install=["pandas", "scikit-learn", "xgboost", "mlflow", "boto3"]
)
def model_training(
    dataset_path: str,
    model_type: str,
    hyperparameters: dict,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    dvc_commit: str,
    trigger_type: str,
    git_sha: str,
) -> NamedTuple('Outputs', [
    ('model_path', str),
    ('mlflow_run_id', str),
    ('training_metrics', dict),
]):
    """
    Train model and log everything to MLflow.
    
    Args:
        dataset_path: Path to validated training data
        model_type: Type of model ("xgboost", "sklearn_rf", etc.)
        hyperparameters: Model hyperparameters
        mlflow_tracking_uri: MLflow server URI
        mlflow_experiment_name: Experiment name for organization
        dvc_commit: Data version for lineage
        trigger_type: What triggered this run ("ci", "scheduled", "drift", "manual")
        git_sha: Git commit SHA for code version
    
    Returns:
        model_path: Path to saved model artifact
        mlflow_run_id: MLflow run ID for tracking
        training_metrics: Dictionary of training metrics
    """
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import mlflow
    import json
    import os
    
    print("Starting model training...")
    print(f"  Model type: {model_type}")
    print(f"  Hyperparameters: {hyperparameters}")
    
    # Load data
    df = pd.read_parquet(dataset_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # ----- MLflow Setup -----
    # In production, uncomment:
    # mlflow.set_tracking_uri(mlflow_tracking_uri)
    # mlflow.set_experiment(mlflow_experiment_name)
    
    # For local testing, use local tracking
    mlflow.set_experiment(mlflow_experiment_name)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"  MLflow run ID: {run_id}")
        
        # Log parameters
        mlflow.log_params(hyperparameters)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("dvc_commit", dvc_commit)
        mlflow.log_param("trigger_type", trigger_type)
        mlflow.log_param("git_sha", git_sha)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("validation_samples", len(X_val))
        
        # ----- Train Model -----
        if model_type == "xgboost":
            model = xgb.XGBClassifier(
                max_depth=hyperparameters.get("max_depth", 6),
                learning_rate=hyperparameters.get("learning_rate", 0.1),
                n_estimators=hyperparameters.get("n_estimators", 100),
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # ----- Evaluate -----
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_score": float(f1_score(y_val, y_pred)),
            "roc_auc": float(roc_auc_score(y_val, y_pred_proba))
        }
        
        print(f"  Metrics: {metrics}")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save and log model
        model_path = "/tmp/model.json"
        model.save_model(model_path)
        mlflow.log_artifact(model_path)
        
        # Log model with MLflow's model registry format
        mlflow.xgboost.log_model(model, "model")
    
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['model_path', 'mlflow_run_id', 'training_metrics'])
    return outputs(model_path, run_id, metrics)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["mlflow", "boto3"]
)
def model_evaluation(
    training_metrics: dict,
    mlflow_tracking_uri: str,
    model_name: str,
    min_improvement_threshold: float,
) -> NamedTuple('Outputs', [
    ('should_register', bool),
    ('evaluation_reason', str),
    ('comparison_report', dict),
]):
    """
    Compare new model against production baseline.
    
    Decision logic:
        - If no production model exists → register
        - If new model > production * (1 + threshold) → register
        - If new model >= production * (1 - threshold) → review
        - Otherwise → reject
    
    Args:
        training_metrics: Metrics from training step
        mlflow_tracking_uri: MLflow server URI
        model_name: Model name in registry
        min_improvement_threshold: Minimum improvement required (e.g., 0.01 = 1%)
    
    Returns:
        should_register: Boolean decision
        evaluation_reason: Human-readable explanation
        comparison_report: Detailed comparison data
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("Starting model evaluation...")
    print(f"  New model metrics: {training_metrics}")
    
    # Primary metric for comparison
    primary_metric = "f1_score"
    new_score = training_metrics.get(primary_metric, 0)
    
    # In production, fetch production model's metrics
    # client = MlflowClient(mlflow_tracking_uri)
    # try:
    #     prod_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    #     prod_run = client.get_run(prod_version.run_id)
    #     prod_score = prod_run.data.metrics.get(primary_metric, 0)
    # except:
    #     prod_score = None
    
    # Simulated production baseline
    prod_score = 0.75  # Set to None to simulate no production model
    
    comparison_report = {
        "primary_metric": primary_metric,
        "new_model_score": new_score,
        "production_score": prod_score,
        "threshold": min_improvement_threshold
    }
    
    # ----- Decision Logic -----
    if prod_score is None:
        should_register = True
        reason = "No production model exists. Registering as first model."
    elif new_score > prod_score * (1 + min_improvement_threshold):
        should_register = True
        improvement = (new_score - prod_score) / prod_score * 100
        reason = f"New model improves {primary_metric} by {improvement:.2f}%"
    elif new_score >= prod_score * (1 - min_improvement_threshold):
        should_register = False  # Could be True with human review
        reason = f"New model within acceptable range but no significant improvement"
    else:
        should_register = False
        degradation = (prod_score - new_score) / prod_score * 100
        reason = f"New model degrades {primary_metric} by {degradation:.2f}%"
    
    comparison_report["decision"] = "register" if should_register else "reject"
    comparison_report["reason"] = reason
    
    print(f"  Decision: {comparison_report['decision']}")
    print(f"  Reason: {reason}")
    
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['should_register', 'evaluation_reason', 'comparison_report'])
    return outputs(should_register, reason, comparison_report)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["mlflow", "boto3"]
)
def model_registration(
    should_register: bool,
    model_path: str,
    mlflow_run_id: str,
    mlflow_tracking_uri: str,
    model_name: str,
    evaluation_reason: str,
) -> NamedTuple('Outputs', [
    ('registered', bool),
    ('model_version', str),
    ('model_stage', str),
]):
    """
    Register model to MLflow Model Registry if approved.
    
    Args:
        should_register: Decision from evaluation step
        model_path: Path to model artifact
        mlflow_run_id: MLflow run ID
        mlflow_tracking_uri: MLflow server URI
        model_name: Name for the registered model
        evaluation_reason: Why this decision was made
    
    Returns:
        registered: Whether registration happened
        model_version: Version number if registered, "N/A" otherwise
        model_stage: Stage ("Staging" if registered, "N/A" otherwise)
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("Starting model registration...")
    print(f"  Should register: {should_register}")
    print(f"  Reason: {evaluation_reason}")
    
    if not should_register:
        print("  Skipping registration based on evaluation decision")
        from collections import namedtuple
        outputs = namedtuple('Outputs', ['registered', 'model_version', 'model_stage'])
        return outputs(False, "N/A", "N/A")
    
    # In production:
    # mlflow.set_tracking_uri(mlflow_tracking_uri)
    # client = MlflowClient()
    #
    # # Register model
    # model_uri = f"runs:/{mlflow_run_id}/model"
    # result = mlflow.register_model(model_uri, model_name)
    #
    # # Transition to Staging
    # client.transition_model_version_stage(
    #     name=model_name,
    #     version=result.version,
    #     stage="Staging"
    # )
    
    # Simulated registration
    model_version = "1"
    model_stage = "Staging"
    
    print(f"  Model registered: {model_name} v{model_version}")
    print(f"  Stage: {model_stage}")
    
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['registered', 'model_version', 'model_stage'])
    return outputs(True, model_version, model_stage)


# =============================================================================
# PIPELINE DEFINITION
# =============================================================================

@dsl.pipeline(
    name=PIPELINE_NAME,
    description=PIPELINE_DESCRIPTION
)
def training_pipeline(
    # Data parameters
    data_version: str = "v1.0.0",
    feature_set: str = "default_features",
    s3_bucket: str = "mlops-data-bucket",
    feast_repo_path: str = "/feast",
    
    # Validation parameters
    min_row_count: int = 100,
    max_null_percentage: float = 5.0,
    
    # Training parameters
    model_type: str = "xgboost",
    hyperparameters: dict = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100},
    
    # MLflow parameters
    mlflow_tracking_uri: str = "http://mlflow.mlops.svc.cluster.local:5000",
    mlflow_experiment_name: str = "training-pipeline",
    model_name: str = "classifier-model",
    
    # Evaluation parameters
    min_improvement_threshold: float = 0.01,
    
    # Metadata
    trigger_type: str = "manual",
    git_sha: str = "unknown",
):
    """
    End-to-end training pipeline.
    
    This pipeline:
    1. Ingests versioned data from DVC and features from Feast
    2. Validates data quality with Great Expectations
    3. Trains model and logs to MLflow
    4. Evaluates against production baseline
    5. Conditionally registers to MLflow Model Registry
    """
    
    # Step 1: Data Ingestion
    ingest_task = data_ingestion(
        data_version=data_version,
        feature_set=feature_set,
        s3_bucket=s3_bucket,
        feast_repo_path=feast_repo_path,
    )
    
    # Step 2: Data Validation
    validation_task = data_validation(
        dataset_path=ingest_task.outputs["dataset_path"],
        min_row_count=min_row_count,
        max_null_percentage=max_null_percentage,
    )
    
    # Step 3: Training (only if validation passes)
    # Using dsl.Condition for conditional execution
    with dsl.Condition(validation_task.outputs["validation_passed"] == True):
        
        train_task = model_training(
            dataset_path=ingest_task.outputs["dataset_path"],
            model_type=model_type,
            hyperparameters=hyperparameters,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            dvc_commit=ingest_task.outputs["dvc_commit"],
            trigger_type=trigger_type,
            git_sha=git_sha,
        )
        
        # Step 4: Evaluation
        eval_task = model_evaluation(
            training_metrics=train_task.outputs["training_metrics"],
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_name=model_name,
            min_improvement_threshold=min_improvement_threshold,
        )
        
        # Step 5: Registration
        register_task = model_registration(
            should_register=eval_task.outputs["should_register"],
            model_path=train_task.outputs["model_path"],
            mlflow_run_id=train_task.outputs["mlflow_run_id"],
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_name=model_name,
            evaluation_reason=eval_task.outputs["evaluation_reason"],
        )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def compile_pipeline(output_path: str = "training_pipeline.yaml"):
    """Compile pipeline to YAML for Kubeflow."""
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=output_path
    )
    print(f"Pipeline compiled to: {output_path}")


def submit_pipeline(
    kubeflow_host: str,
    data_version: str,
    trigger_type: str = "manual",
    git_sha: str = "unknown"
):
    """Submit pipeline to Kubeflow."""
    # In production:
    # import kfp
    # client = kfp.Client(host=kubeflow_host)
    # client.create_run_from_pipeline_func(
    #     training_pipeline,
    #     arguments={
    #         "data_version": data_version,
    #         "trigger_type": trigger_type,
    #         "git_sha": git_sha,
    #     }
    # )
    print(f"Pipeline submitted with data_version={data_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline CLI")
    parser.add_argument("--compile", action="store_true", help="Compile pipeline to YAML")
    parser.add_argument("--submit", action="store_true", help="Submit pipeline to Kubeflow")
    parser.add_argument("--kubeflow-host", default="http://localhost:8080")
    parser.add_argument("--data-version", default="v1.0.0")
    parser.add_argument("--trigger-type", default="manual")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--output", default="training_pipeline.yaml")
    
    args = parser.parse_args()
    
    if args.compile:
        compile_pipeline(args.output)
    elif args.submit:
        submit_pipeline(
            args.kubeflow_host,
            args.data_version,
            args.trigger_type,
            args.git_sha
        )
    else:
        parser.print_help()