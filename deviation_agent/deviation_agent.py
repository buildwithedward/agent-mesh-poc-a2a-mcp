import json
import boto3
import pickle
import pandas as pd
import numpy as np
import io
import urllib.parse
import traceback

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


def deviation_agent(dataset_bucket, dataset_key):

    batch_id = dataset_key.split("/")[-1].replace(".csv", "")

    # Load model
    model_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    artifact = pickle.loads(model_obj["Body"].read())

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_idx = artifact["feature_names"]
    threshold = float(artifact["threshold"])

    # Load dataset
    data_obj = s3.get_object(Bucket=dataset_bucket, Key=dataset_key)
    X = pd.read_csv(io.BytesIO(data_obj["Body"].read()), header=None)

    X = X.iloc[:, feature_idx]
    X = X.fillna(X.median())
    X_scaled = scaler.transform(X)

    # Detect anomalies
    scores = model.decision_function(X_scaled)
    anomalous_idxs = np.where(scores < threshold)[0]

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

    # Save output
    deviation_key = f"{OUTPUT_PREFIX}{batch_id}.json"

    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=deviation_key,
        Body=json.dumps(results, indent=2),
        ContentType="application/json"
    )

    # Emit EventBridge event
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
                    "deviation_key": deviation_key
                }),
                "EventBusName": "default"
            }
        ]
    )

    return deviation_key


def lambda_handler(event, context):

    try:
        record = event["Records"][0]

        dataset_bucket = record["s3"]["bucket"]["name"]
        dataset_key = urllib.parse.unquote_plus(
            record["s3"]["object"]["key"]
        )

        print(f"Triggered for: s3://{dataset_bucket}/{dataset_key}")

        if dataset_key.startswith(OUTPUT_PREFIX):
            return {"status": "SKIPPED_OUTPUT_FILE"}

        deviation_key = deviation_agent(dataset_bucket, dataset_key)

        return {
            "status": "SUCCESS",
            "deviation_s3_uri": f"s3://{OUTPUT_BUCKET}/{deviation_key}"
        }

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {"status": "FAILED", "error": str(e)}
