import os
import logging
import traceback
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------------------------------
# FastMCP Server — RCA Agent
# AgentCore expects: POST /mcp on port 8000
# stateless_http=True required (AgentCore manages sessions via Mcp-Session-Id header)
# -------------------------------------------------
mcp = FastMCP("rca-agent", host="0.0.0.0", stateless_http=True)


@mcp.tool()
def run_rca_analysis(
    dataset_bucket: str,
    dataset_key: str,
    model_bucket: str,
    model_key: str,
    deviation_bucket: str,
    deviation_key: str
) -> dict:
    """
    Run Root Cause Analysis (RCA) on anomalous rows identified by the Deviation Agent.

    Uses SHAP (SHapley Additive exPlanations) values from the Isolation Forest model
    to identify the top contributing sensors for each anomalous row, then categorizes
    deviations using z-scores computed among anomalous samples. Results are saved to S3.

    Args:
        dataset_bucket: S3 bucket name containing the input dataset CSV
        dataset_key: S3 object key of the input dataset CSV
        model_bucket: S3 bucket name containing the Isolation Forest model artifact
        model_key: S3 object key of the model pickle file
        deviation_bucket: S3 bucket name containing the Deviation Agent output JSON
        deviation_key: S3 object key of the Deviation Agent output JSON

    Returns:
        Dictionary containing:
        - status: "SUCCESS" or "FAILED"
        - batch_id: Identifier of the processed batch (on success)
        - rca_output_s3_uri: S3 URI of the RCA output JSON (on success)
        - error: Error message string (on failure)
    """
    batch_id = deviation_key.split("/")[-1].replace(".json", "")
    try:
        from agent_logic import run_rca, write_status

        write_status(batch_id, "RUNNING")

        result = run_rca(
            dataset_bucket,
            dataset_key,
            model_bucket,
            model_key,
            deviation_bucket,
            deviation_key
        )

        rca_output_s3_uri = f"s3://ag-agent-mesh/RCA_Agent_Output/{result['batch_id']}.json"

        write_status(
            batch_id,
            "SUCCESS",
            output_path=rca_output_s3_uri
        )

        return {
            "status": "SUCCESS",
            "batch_id": result["batch_id"],
            "rca_output_s3_uri": rca_output_s3_uri
        }

    except Exception as e:
        write_status(batch_id, "FAILED", message=str(e))
        return {
            "status": "FAILED",
            "batch_id": batch_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
