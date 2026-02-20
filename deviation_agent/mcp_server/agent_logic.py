import json
import logging
import boto3
import pickle
import pandas as pd
import numpy as np
import io
import os
import traceback

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Disable joblib multiprocessing
# -------------------------------------------------
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_BUCKET = "ag-agent-mesh"
MODEL_KEY = "Isolation_Forest_Model/isolation_forest_secom.pkl"

OUTPUT_BUCKET = "ag-agent-mesh"
OUTPUT_PREFIX = "Deviation_Agent_Output/"

# -------------------------------------------------
# AWS Client
# -------------------------------------------------
s3 = boto3.client("s3")


# -------------------------------------------------
# Core Deviation Logic
# -------------------------------------------------
def deviation_agent(dataset_bucket: str, dataset_key: str) -> str:
    """
    Run Isolation Forest anomaly detection on a dataset.

    Args:
        dataset_bucket: S3 bucket containing the input dataset CSV
        dataset_key: S3 key of the input dataset CSV file

    Returns:
        S3 key of the deviation output JSON file
    """
    logger.info(f"Loading dataset from s3://{dataset_bucket}/{dataset_key}")

    batch_id = dataset_key.split("/")[-1].replace(".json", "").replace(".csv", "")

    # -------------------------------------------------
    # Load Model Artifact
    # -------------------------------------------------
    model_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    artifact = pickle.loads(model_obj["Body"].read())

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_idx = artifact["feature_names"]
    threshold = float(artifact["threshold"])

    try:
        model.set_params(n_jobs=1)
    except Exception:
        pass

    logger.info("Model loaded successfully")

    # -------------------------------------------------
    # Load Dataset (CSV)
    # -------------------------------------------------
    data_obj = s3.get_object(Bucket=dataset_bucket, Key=dataset_key)
    X = pd.read_csv(io.BytesIO(data_obj["Body"].read()), header=None)

    X = X.iloc[:, feature_idx]
    X = X.fillna(X.median())

    X_scaled = scaler.transform(X)

    logger.info("Data preprocessing completed")

    # -------------------------------------------------
    # Detect Anomalies
    # -------------------------------------------------
    scores = model.decision_function(X_scaled)
    anomalous_idxs = np.where(scores < threshold)[0]

    logger.info(f"Detected {len(anomalous_idxs)} anomalies")

    results = []

    for idx in anomalous_idxs:
        results.append({
            "batch_id": batch_id,
            "row_index": int(idx),
            "deviation_detected": True,
            "anomaly_score": float(scores[idx]),
            "threshold": threshold,
            "decision": "ANOMALY"
        })

    # -------------------------------------------------
    # Save Output to S3
    # -------------------------------------------------
    deviation_key = f"{OUTPUT_PREFIX}{batch_id}.json"

    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=deviation_key,
        Body=json.dumps(results, indent=2),
        ContentType="application/json"
    )

    logger.info(f"Deviation output saved to s3://{OUTPUT_BUCKET}/{deviation_key}")

    return deviation_key
