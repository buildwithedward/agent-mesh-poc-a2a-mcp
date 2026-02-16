import json
import boto3
import traceback
from service import run_rca

eventbridge = boto3.client("events")
s3 = boto3.client("s3")

# Hardcoded RCA output bucket
RCA_BUCKET = "ag-agent-mesh"
RCA_FOLDER = "RCA_Agent_Output"
STATUS_FOLDER = "status"

def write_status(batch_id, status, message=None, output_path=None):
    status_key = f"{STATUS_FOLDER}/{batch_id}.json"

    body = {
        "batch_id": batch_id,
        "status": status,
        "message": message,
        "output_path": output_path
    }

    s3.put_object(
        Bucket=RCA_BUCKET,
        Key=status_key,
        Body=json.dumps(body),
        ContentType="application/json"
    )

def lambda_handler(event, context):

    detail = event["detail"]

    dataset_bucket = detail["dataset_bucket"]
    dataset_key = detail["dataset_key"]

    model_bucket = detail["model_bucket"]
    model_key = detail["model_key"]

    deviation_bucket = detail["deviation_bucket"]
    deviation_key = detail["deviation_key"]

    batch_id = detail["batch_id"]

    print(f"Request ID: {context.aws_request_id}")
    print(f"Processing batch: {batch_id}")

    try:
        # Step 1: Write RUNNING status
        write_status(batch_id, "RUNNING")

        # Step 2: Run RCA
        result = run_rca(
            dataset_bucket,
            dataset_key,
            model_bucket,
            model_key,
            deviation_bucket,
            deviation_key
        )

        rca_key = f"{RCA_FOLDER}/{batch_id}.json"
        output_path = f"s3://{RCA_BUCKET}/{rca_key}"

        # Step 3: Write SUCCESS status
        write_status(
            batch_id,
            "SUCCESS",
            output_path=output_path
        )

        # Step 4: Publish success event to CAPA
        eventbridge.put_events(
            Entries=[
                {
                    "Source": "agilsium.rca.agent",
                    "DetailType": "RCACompleted",
                    "Detail": json.dumps({
                        "batch_id": batch_id,
                        "rca_bucket": RCA_BUCKET,
                        "rca_key": rca_key,
                        "status": "SUCCESS"
                    }),
                    "EventBusName": "default"
                }
            ]
        )

        return {"status": "SUCCESS"}

    except Exception as e:

        error_message = str(e)
        stack_trace = traceback.format_exc()

        print("RCA FAILED")
        print(error_message)
        print(stack_trace)
        print(f"Request ID: {context.aws_request_id}")

        write_status(
            batch_id,
            "FAILED",
            message=error_message
        )

        eventbridge.put_events(
            Entries=[
                {
                    "Source": "agilsium.rca.agent",
                    "DetailType": "RCAFailed",
                    "Detail": json.dumps({
                        "batch_id": batch_id,
                        "status": "FAILED",
                        "error": error_message
                    }),
                    "EventBusName": "default"
                }
            ]
        )

        raise e
