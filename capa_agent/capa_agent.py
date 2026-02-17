import json
import os
import boto3
import random
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ============================================================
# AWS Client
# ============================================================

s3 = boto3.client("s3")

# ============================================================
# Environment Variables (Set in Lambda)
# ============================================================

QDRANT_URL = os.environ["VDB_URL"]
QDRANT_API_KEY = os.environ["VDB_API"]

COLLECTION_NAME = "capa_knowledge_base"
OUTPUT_BUCKET = "ag-agent-mesh"
OUTPUT_PREFIX = "CAPA_Agent_Output"

# ============================================================
# Qdrant Client
# ============================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ============================================================
# CAPA Agent Core Logic
# ============================================================

def capa_agent(bucket, key):

    print(f"Reading RCA from s3://{bucket}/{key}")

    # ---------------------------------------------------
    # 1️⃣ Read RCA Output
    # ---------------------------------------------------
    response = s3.get_object(Bucket=bucket, Key=key)
    rca_data = json.loads(response["Body"].read())

    batch_id = rca_data["batch_id"]
    rca_results = rca_data["rca_results"]

    CATEGORY_MAP = {
        "Large deviation among anomalies": "L",
        "Moderate deviation among anomalies": "M",
        "Within anomaly variation": "N"
    }

    sensor_level_capa = []

    # ---------------------------------------------------
    # 2️⃣ Process ALL Root Causes
    # ---------------------------------------------------
    for result in rca_results:

        row_index = result["row_index"]

        for root in result["root_causes"]:

            sensor_id = root["sensor"]
            z_category = root["z_category"]

            if z_category not in CATEGORY_MAP:
                continue

            severity_type = CATEGORY_MAP[z_category]

            # ---------------------------------------------------
            # 3️⃣ Fetch CAPAs from Qdrant by severity_type
            # ---------------------------------------------------
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="severity_type",
                        match=MatchValue(value=severity_type)
                    )
                ]
            )

            points, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=50
            )

            matching_capas = [p.payload for p in points]

            capa_ids = []
            capa_details = []

            # ---------------------------------------------------
            # 4️⃣ Probabilistic CAPA Selection
            # ---------------------------------------------------

            if severity_type == "L":
                num_to_pick = random.randint(0, 2)

            elif severity_type == "M":
                num_to_pick = random.randint(0, 1)

            else:  # N
                num_to_pick = random.randint(0, 1)

            if num_to_pick > 0 and matching_capas:
                selected = random.sample(
                    matching_capas,
                    min(num_to_pick, len(matching_capas))
                )
            else:
                selected = []

            # ---------------------------------------------------
            # 5️⃣ Build Output Based on Severity
            # ---------------------------------------------------

            for payload in selected:

                capa_id = payload.get("capa_id")

                if severity_type in ["L", "M"]:

                    capa_ids.append(capa_id)

                    capa_details.append({
                        "capa_id": capa_id,
                        "statistical_interpretation": payload.get("statistical_interpretation"),
                        "context_description": payload.get("context_description"),
                        "probable_cause_description": payload.get("probable_cause_description"),
                        "corrective_actions": payload.get("corrective_actions", []),
                        "preventive_actions": payload.get("preventive_actions", [])
                    })

                else:  # N severity
                    preventive = payload.get("preventive_actions", [])
                    capa_details.append({
                        "preventive_actions": preventive[:1]
                    })

            sensor_level_capa.append({
                "row_index": row_index,
                "sensor": sensor_id,
                "z_category": z_category,
                "severity_type": severity_type,
                "capa_ids": capa_ids,
                "capa_details": capa_details
            })

    # ---------------------------------------------------
    # 6️⃣ Build Final Output
    # ---------------------------------------------------
    output = {
        "batch_id": batch_id,
        "sensor_level_capa": sensor_level_capa
    }

    output_key = f"{OUTPUT_PREFIX}/{batch_id}_capa_output.json"

    # ---------------------------------------------------
    # 7️⃣ Save to S3
    # ---------------------------------------------------
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=output_key,
        Body=json.dumps(output, indent=2),
        ContentType="application/json"
    )

    print(f"CAPA output saved to s3://{OUTPUT_BUCKET}/{output_key}")

    return {
        "status": "SUCCESS",
        "batch_id": batch_id,
        "capa_output_s3_uri": f"s3://{OUTPUT_BUCKET}/{output_key}"
    }


# ============================================================
# Lambda Entry Point
# ============================================================

def lambda_handler(event, context):

    try:
        print("Received Event:")
        print(json.dumps(event, indent=2))

        detail = event["Detail"]

        if detail["status"] != "SUCCESS":
            print("RCA status not SUCCESS. Skipping.")
            return {"status": "SKIPPED"}

        bucket = detail["rca_bucket"]
        key = detail["rca_key"]

        return capa_agent(bucket, key)

    except Exception as e:

        print("Error occurred:")
        print(str(e))

        return {
            "status": "FAILED",
            "error": str(e)
        }
