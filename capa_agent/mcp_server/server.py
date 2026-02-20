import os
import logging
import traceback
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------------------------------
# FastMCP Server — CAPA Agent
# AgentCore expects: POST /mcp on port 8000
# stateless_http=True required (AgentCore manages sessions via Mcp-Session-Id header)
# -------------------------------------------------
mcp = FastMCP("capa-agent", host="0.0.0.0", stateless_http=True)


@mcp.tool()
def run_capa_selection(rca_bucket: str, rca_key: str) -> dict:
    """
    Select Corrective and Preventive Actions (CAPAs) for anomalies identified in RCA output.

    Reads the RCA Agent output from S3, maps each root cause sensor's z-category to a
    severity level (Large → L, Moderate → M, Within → N), queries the Qdrant vector
    database for matching CAPAs, applies probabilistic selection, and saves the results
    to S3. Requires VDB_URL and VDB_API environment variables for Qdrant access.

    Args:
        rca_bucket: S3 bucket name containing the RCA Agent output JSON
        rca_key: S3 object key of the RCA Agent output JSON

    Returns:
        Dictionary containing:
        - status: "SUCCESS" or "FAILED"
        - batch_id: Identifier of the processed batch (on success)
        - capa_output_s3_uri: S3 URI of the CAPA output JSON (on success)
        - error: Error message string (on failure)
    """
    try:
        from agent_logic import capa_agent
        return capa_agent(rca_bucket, rca_key)
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
