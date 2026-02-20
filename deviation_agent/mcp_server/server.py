import os
import logging
import traceback
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------------------------------
# FastMCP Server — Deviation Agent
# AgentCore expects: POST /mcp on port 8000
# stateless_http=True required (AgentCore manages sessions via Mcp-Session-Id header)
# -------------------------------------------------
mcp = FastMCP("deviation-agent", host="0.0.0.0", stateless_http=True)


@mcp.tool()
def run_deviation_detection(dataset_bucket: str, dataset_key: str) -> dict:
    """
    Run Isolation Forest anomaly detection on a manufacturing dataset stored in S3.

    Loads a pre-trained Isolation Forest model from S3, applies it to the input
    dataset, identifies anomalous rows, and saves the results back to S3.

    Args:
        dataset_bucket: S3 bucket name containing the input dataset CSV file
        dataset_key: S3 object key of the input dataset CSV file

    Returns:
        Dictionary containing:
        - status: "SUCCESS" or "FAILED"
        - deviation_s3_uri: S3 URI of the deviation output JSON (on success)
        - error: Error message string (on failure)
    """
    try:
        from agent_logic import deviation_agent
        deviation_key = deviation_agent(dataset_bucket, dataset_key)
        return {
            "status": "SUCCESS",
            "deviation_s3_uri": f"s3://ag-agent-mesh/{deviation_key}"
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
