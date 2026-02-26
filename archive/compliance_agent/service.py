import boto3
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================

REGION = "us-west-2"

BUCKET_NAME = "ag-agent-mesh"
INPUT_KEY = "CAPA_Agent_Output/secom_567_data_capa_output.json"
OUTPUT_FOLDER = "Compliance_Agent_Output/"

MODEL_ID = "meta.llama3-1-8b-instruct-v1:0"
MAX_CONCURRENT_CALLS = 5

# ==========================================================
# AWS CLIENTS
# ==========================================================

def get_s3_client():
    return boto3.client("s3", region_name=REGION)

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=REGION)

# ==========================================================
# FETCH CAPA OUTPUT
# ==========================================================

def fetch_capa_output():
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=INPUT_KEY)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except Exception as e:
        print(f"Error reading from S3: {e}")
        raise

# ==========================================================
# SEVERITY MAPPING
# ==========================================================

def map_severity(z_category):
    z = z_category.lower()
    if "large" in z:
        return "High"
    elif "moderate" in z:
        return "Medium"
    elif "within" in z:
        return "Low"
    return "Unknown"

# ==========================================================
# COMPLIANCE RULE ENGINE (DETERMINISTIC)
# ==========================================================

def evaluate_compliance(sensor):
    z = sensor.get("z_category", "").lower()
    capa_details = sensor.get("capa_details", [])

    corrective_count = 0
    preventive_count = 0

    for capa in capa_details:
        corrective_count += len(capa.get("corrective_actions", []))
        preventive_count += len(capa.get("preventive_actions", []))

    if not capa_details:
        return "FAIL"

    if "large" in z:
        return "PASS" if corrective_count >= 1 and preventive_count >= 1 else "FAIL"
    elif "moderate" in z:
        return "PASS" if preventive_count >= 1 and corrective_count <= 1 else "FAIL"
    elif "within" in z:
        return "PASS" if corrective_count == 0 else "FAIL"

    return "UNKNOWN"

# ==========================================================
# SAFE BEDROCK CALL (LLAMA PROMPT FORMAT)
# ==========================================================

def call_bedrock(prompt_text):
    client = get_bedrock_client()

    # Meta Llama 3 Instruct expects proper special tokens
    llama3_prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    body = {
        "prompt": llama3_prompt,
        "max_gen_len": 600,
        "temperature": 0.2,
        "top_p": 0.9
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response["body"].read())
    text_output = response_body.get("generation", "").strip()

    if not text_output:
        raise ValueError("Model returned empty response.")

    # Remove markdown fences if present
    if text_output.startswith("```"):
        text_output = text_output.replace("```json", "").replace("```", "").strip()

    # --- Robust JSON Extraction ---
    decoder = json.JSONDecoder()

    try:
        obj, _ = decoder.raw_decode(text_output)
        return obj
    except json.JSONDecodeError:
        pass

    for i in range(len(text_output)):
        try:
            obj, _ = decoder.raw_decode(text_output[i:])
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse valid JSON from model output:\n{text_output}")

# ==========================================================
# ASYNC SENSOR PROCESSOR
# ==========================================================

async def process_sensor(sensor, semaphore, executor):
    async with semaphore:
        loop = asyncio.get_running_loop()

        severity = map_severity(sensor.get("z_category", ""))
        compliance_status = evaluate_compliance(sensor)

        if not sensor.get("capa_details"):
            return {
                "sensor": sensor["sensor"],
                "severity_classification": severity,
                "compliance_status": "FAIL",
                "failure_reason": "No preventive or corrective actions defined.",
                "recommendation": "Review anomaly and assign appropriate CAPA."
            }

        preventive = []
        corrective = []

        for capa in sensor["capa_details"]:
            preventive.extend(capa.get("preventive_actions", []))
            corrective.extend(capa.get("corrective_actions", []))

        # Fallback if actions somehow parse empty
        preventive_text = "\n".join(f"- {p}" for p in preventive) if preventive else "None"
        corrective_text = "\n".join(f"- {c}" for c in corrective) if corrective else "None"

        prompt = f"""You are a regulatory compliance summarization engine.
Your task: Summarize the provided Preventive Actions and Corrective Actions.

Rules:
- Produce ONLY valid JSON.
- Do NOT include explanations.
- Do NOT include markdown.
- Output must be a single JSON object.

Output Schema:
{{
  "failure_reason": "A concise summary (1-2 sentences) of the preventive controls.",
  "recommendation": "A concise summary (1-2 sentences) of the corrective actions."
}}

Preventive Actions:
{preventive_text}

Corrective Actions:
{corrective_text}

Return ONLY the JSON object.
"""

        try:
            result = await loop.run_in_executor(executor, call_bedrock, prompt)
        except Exception as e:
            print(f"Bedrock call failed for sensor {sensor['sensor']}: {e}")
            result = {"failure_reason": "Error generating summary", "recommendation": "Error generating recommendation"}

        return {
            "sensor": sensor["sensor"],
            "severity_classification": severity,
            "compliance_status": compliance_status,
            "failure_reason": result.get("failure_reason", ""),
            "recommendation": result.get("recommendation", "")
        }

# ==========================================================
# MAIN ASYNC ENGINE
# ==========================================================

async def run_compliance_engine():
    print("Fetching data from S3...")
    data = fetch_capa_output()

    rows = {}
    for sensor in data.get("sensor_level_capa", []):
        rows.setdefault(sensor["row_index"], []).append(sensor)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS)

    final_rows = []
    
    print(f"Processing {len(data.get('sensor_level_capa', []))} sensors across {len(rows)} rows...")

    for row_id, sensors in rows.items():
        tasks = [process_sensor(sensor, semaphore, executor) for sensor in sensors]
        sensor_results = await asyncio.gather(*tasks)

        final_rows.append({
            "row_index": row_id,
            "sensors": sensor_results
        })

    return final_rows

# ==========================================================
# UPLOAD RESULT TO S3
# ==========================================================

def upload_output(result):
    s3 = get_s3_client()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_key = f"{OUTPUT_FOLDER}compliance_output_{timestamp}.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=output_key,
        Body=json.dumps(result, indent=4),
        ContentType="application/json"
    )

    print(f"Successfully uploaded to s3://{BUCKET_NAME}/{output_key}")

# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":
    final_result = asyncio.run(run_compliance_engine())
    upload_output(final_result)