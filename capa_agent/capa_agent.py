import json
import os
import boto3
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range

# ============================================================
# AWS Clients
# ============================================================

s3 = boto3.client("s3")
eventbridge = boto3.client("events")

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
    """
    Reads RCA JSON from S3,
    fetches matching CAPAs from Qdrant,
    stores CAPA output to S3,
    emits EventBridge event.
    """

    print(f"Reading RCA from s3://{bucket}/{key}")

    # ---------------------------------------------------
    # 1️⃣ Read RCA Output
    # ---------------------------------------------------
    response = s3.get_object(Bucket=bucket, Key=key)
    rca_data = json.loads(response["Body"].read())

    batch_id = rca_data["batch_id"]
    rca_results = rca_data["rca_results"]

    CATEGORY_MAP = {
        "Large deviation among anomalies": 3,
        "Moderate deviation among anomalies": 2,
        "Within anomaly variation": 1
    }

    sensor_level_capa = []

    # ---------------------------------------------------
    # 2️⃣ Process RCA Results
    # ---------------------------------------------------
    for result in rca_results:

        row_index = result["row_index"]

        for root in result["root_causes"]:

            sensor_id = root["sensor"]
            z_category = root["z_category"]

            if z_category not in CATEGORY_MAP:
                continue

            severity = CATEGORY_MAP[z_category]

            # ---------------------------------------------------
            # 3️⃣ Fetch CAPAs from Qdrant
            # ---------------------------------------------------
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="severity_score",
                        range=Range(gte=severity)
                    )
                ]
            )

            points, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=100
            )

            capa_ids = []
            capa_details = []

            for point in points:
                payload = point.payload

                capa_id = payload.get("capa_id")
                corrective_action = payload.get("corrective_action")
                preventive_action = payload.get("preventive_action")

                capa_ids.append(capa_id)

                capa_details.append({
                    "capa_id": capa_id,
                    "corrective_actions": [corrective_action],
                    "preventive_actions": [preventive_action]
                })

            sensor_level_capa.append({
                "row_index": row_index,
                "sensor": sensor_id,
                "z_category": z_category,
                "capa_ids": capa_ids,
                "capa_details": capa_details
            })

    # ---------------------------------------------------
    # 4️⃣ Build Final Output
    # ---------------------------------------------------
    output = {
        "batch_id": batch_id,
        "sensor_level_capa": sensor_level_capa
    }

    output_key = f"{OUTPUT_PREFIX}/{batch_id}_capa_output.json"

    # ---------------------------------------------------
    # 5️⃣ Save to S3
    # ---------------------------------------------------
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=output_key,
        Body=json.dumps(output, indent=2),
        ContentType="application/json"
    )

    print(f"CAPA output saved to s3://{OUTPUT_BUCKET}/{output_key}")

    # ---------------------------------------------------
    # 6️⃣ Emit EventBridge Success Event
    # ---------------------------------------------------
    eventbridge.put_events(
        Entries=[
            {
                "Source": "agilsium.capa.agent",
                "DetailType": "CAPACompleted",
                "Detail": json.dumps({
                    "batch_id": batch_id,
                    "capa_bucket": OUTPUT_BUCKET,
                    "capa_key": output_key,
                    "status": "SUCCESS"
                }),
                "EventBusName": "default"
            }
        ]
    )

    print("CAPACompleted event emitted.")

    return {
        "status": "SUCCESS",
        "batch_id": batch_id,
        "capa_output_s3_uri": f"s3://{OUTPUT_BUCKET}/{output_key}"
    }


# ============================================================
# Lambda Entry Point (EventBridge Trigger)
# ============================================================

def lambda_handler(event, context):

    try:
        print("Received EventBridge Event:")
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

        # ---------------------------------------------------
        # Emit Failure Event
        # ---------------------------------------------------
        eventbridge.put_events(
            Entries=[
                {
                    "Source": "agilsium.capa.agent",
                    "DetailType": "CAPAFailed",
                    "Detail": json.dumps({
                        "error": str(e),
                        "status": "FAILED"
                    }),
                    "EventBusName": "default"
                }
            ]
        )

        return {
            "status": "FAILED",
            "error": str(e)
        }
