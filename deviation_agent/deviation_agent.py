import json
import boto3
import pickle
import pandas as pd
import numpy as np
import io
import urllib.parse
import traceback
import os

# -------------------------------------------------
# Disable joblib multiprocessing in Lambda
# -------------------------------------------------
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_BUCKET = "ag-agent-mesh"
MODEL_KEY = "Isolation_Forest_Model/isolation_forest_secom.pkl"

OUTPUT_BUCKET = "ag-agent-mesh"
OUTPUT_PREFIX = "Deviation_Agent_Output/"

EVENT_SOURCE = "agent.deviation"
EVENT_DETAIL_TYPE = "DeviationCompleted"

# -------------------------------------------------
# AWS Clients
# -------------------------------------------------
s3 = boto3.client("s3")
eventbridge = boto3.client("events")

# -------------------------------------------------
# Core Deviation Logic
# -------------------------------------------------
def deviation_agent(dataset_bucket, dataset_key):

    print(f"Loading dataset from s3://{dataset_bucket}/{dataset_key}")

    batch_id = dataset_key.split("/")[-1].replace(".csv", "")

    # -------------------------------------------------
    # Load Model Artifact
    # -------------------------------------------------
    model_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    artifact = pickle.loads(model_obj["Body"].read())

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_idx = artifact["feature_names"]
    threshold = float(artifact["threshold"])

    # Force single-thread execution
    try:
        model.set_params(n_jobs=1)
    except Exception:
        pass

    print("Model loaded successfully")

    # -------------------------------------------------
    # Load Dataset
    # -------------------------------------------------
    data_obj = s3.get_object(Bucket=dataset_bucket, Key=dataset_key)
    X = pd.read_csv(io.BytesIO(data_obj["Body"].read()), header=None)

    X = X.iloc[:, feature_idx]
    X = X.fillna(X.median())

    X_scaled = scaler.transform(X)

    print("Data preprocessing completed")

    # -------------------------------------------------
    # Detect Anomalies
    # -------------------------------------------------
    scores = model.decision_function(X_scaled)
    anomalous_idxs = np.where(scores < threshold)[0]

    print(f"Detected {len(anomalous_idxs)} anomalies")

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

    print(f"Deviation output saved to s3://{OUTPUT_BUCKET}/{deviation_key}")

    # -------------------------------------------------
    # Emit EventBridge Event
    # -------------------------------------------------
    eventbridge.put_events(
        Entries=[
            {
                "Source": EVENT_SOURCE,
                "DetailType": EVENT_DETAIL_TYPE,
                "Detail": json.dumps({
                    "dataset_bucket": dataset_bucket,
                    "dataset_key": dataset_key,
                    "model_bucket": MODEL_BUCKET,
                    "model_key": MODEL_KEY,
                    "deviation_bucket": OUTPUT_BUCKET,
                    "deviation_key": deviation_key,
                    "status": "SUCCESS"
                }),
                "EventBusName": "default"
            }
        ]
    )

    print("EventBridge event emitted successfully")

    return deviation_key


# -------------------------------------------------
# Lambda Entry Point
# -------------------------------------------------
def lambda_handler(event, context):

    try:
        print("Received event:")
        print(json.dumps(event, indent=2))

        record = event["Records"][0]

        dataset_bucket = record["s3"]["bucket"]["name"]
        dataset_key = urllib.parse.unquote_plus(
            record["s3"]["object"]["key"]
        )

        print(f"Triggered for: s3://{dataset_bucket}/{dataset_key}")

        # Prevent infinite loop (ignore own outputs)
        if dataset_key.startswith(OUTPUT_PREFIX):
            print("Skipping output file trigger")
            return {"status": "SKIPPED_OUTPUT_FILE"}

        deviation_key = deviation_agent(dataset_bucket, dataset_key)

        return {
            "status": "SUCCESS",
            "deviation_s3_uri": f"s3://{OUTPUT_BUCKET}/{deviation_key}"
        }

    except Exception as e:
        print("Error occurred:")
        print(str(e))
        print(traceback.format_exc())

        return {
            "status": "FAILED",
            "error": str(e)
        }
