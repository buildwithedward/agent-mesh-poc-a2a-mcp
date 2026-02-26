import boto3
import pandas as pd
import numpy as np
import io
import json
import pickle
import shap
from sklearn.ensemble import IsolationForest

s3 = boto3.client("s3")

def run_rca(dataset_bucket,
            dataset_key,
            model_bucket,
            model_key,
            deviation_bucket,
            deviation_key):

    # ===============================
    # LOAD DATASET FROM S3
    # ===============================

    obj = s3.get_object(Bucket=dataset_bucket, Key=dataset_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    df.columns = df.columns.astype(int)

    # ===============================
    # LOAD MODEL ARTIFACT
    # ===============================

    obj = s3.get_object(Bucket=model_bucket, Key=model_key)
    artifact = pickle.loads(obj["Body"].read())

    iso_model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    threshold_from_model = artifact["threshold"]

    df = df[feature_names]

    # ===============================
    # LOAD DEVIATION AGENT OUTPUT
    # ===============================

    obj = s3.get_object(Bucket=deviation_bucket, Key=deviation_key)
    deviation_results = json.loads(obj["Body"].read())

    anomalous_rows = [item["row_index"] for item in deviation_results]

    # ===============================
    # SPLIT DATA (Normal / Anomaly)
    # ===============================

    sensor_df = df.copy()
    anomaly_df = sensor_df.loc[anomalous_rows]

    # -------- COMMENTED OUT (Baseline Normal Rows) --------
    # normal_df = sensor_df.drop(index=anomalous_rows)
    # baseline_mean = normal_df.mean()
    # baseline_std = normal_df.std()

    # ===============================
    # COMPUTE ANOMALY-ONLY STATISTICS
    # ===============================

    anomaly_mean = anomaly_df.mean()
    anomaly_std = anomaly_df.std()

    # ===============================
    # SHAP EXPLAINER
    # ===============================

    explainer = shap.TreeExplainer(iso_model)

    TOP_K = 5
    rca_results = []

    # -------- COMMENTED OUT (Baseline Categorization) --------
    # def categorize_baseline_z(z):
    #     if abs(z) >= 3:
    #         return "High (Large deviation from Normal Baseline)"
    #     elif abs(z) >= 1:
    #         return "Medium (Moderate deviation from Historical Range)"
    #     else:
    #         return "Low (Within Normal Variation)"

    def categorize_anomaly_z(z):
        if abs(z) >= 2:
            return "Large deviation among anomalies"
        elif abs(z) >= 1:
            return "Moderate deviation among anomalies"
        else:
            return "Within anomaly variation"

    for item in deviation_results:

        row_id = item["row_index"]
        anomaly_score = item["anomaly_score"]

        row_values = df.loc[row_id, feature_names]
        X_row_scaled = scaler.transform([row_values])

        shap_values = explainer.shap_values(X_row_scaled)[0]
        shap_series = pd.Series(shap_values, index=feature_names)
        top_sensors = shap_series.abs().nlargest(TOP_K)

        causes = []

        for sensor in top_sensors.index:

            value = row_values[sensor]

            # -------- COMMENTED OUT (Baseline Z-score) --------
            # b_mean = baseline_mean[sensor]
            # b_std = baseline_std[sensor]
            # z_score_1 = 0 if b_std == 0 else (value - b_mean) / b_std
            # z1_category = categorize_baseline_z(z_score_1)

            a_mean = anomaly_mean[sensor]
            a_std = anomaly_std[sensor]

            z_score = 0 if a_std == 0 else (value - a_mean) / a_std
            z_category = categorize_anomaly_z(z_score)

            causes.append({
                "sensor": int(sensor),
                "shap_contribution": round(float(shap_series[sensor]), 4),
                "z_score_among_anomalies": round(float(z_score), 3),
                "z_category": z_category
            })

        rca_results.append({
            "batch_id": item["batch_id"],
            "row_index": int(row_id),
            "anomaly_score": float(anomaly_score),
            "root_causes": causes
        })

    rca_output = {
        "batch_id": deviation_results[0]["batch_id"],
        "rca_results": rca_results
    }

    # Hardcoded bucket and folder
    RCA_BUCKET = "ag-agent-mesh"
    RCA_FOLDER = "RCA_Agent_Output"

    rca_key = f"{RCA_FOLDER}/{deviation_results[0]['batch_id']}.json"

    s3.put_object(
        Bucket=RCA_BUCKET,
        Key=rca_key,
        Body=json.dumps(rca_output),
        ContentType="application/json"
    )

    return {
        "batch_id": deviation_results[0]["batch_id"]
    }

