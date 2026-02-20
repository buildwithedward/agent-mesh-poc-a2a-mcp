import os
import logging
import traceback
import asyncio
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------------------------------
# FastMCP Server — Compliance Agent
# AgentCore expects: POST /mcp on port 8000
# stateless_http=True required (AgentCore manages sessions via Mcp-Session-Id header)
# -------------------------------------------------
mcp = FastMCP("compliance-agent", host="0.0.0.0", stateless_http=True)


@mcp.tool()
async def run_compliance_evaluation(capa_bucket: str, capa_key: str) -> dict:
    """
    Run compliance evaluation on CAPA Agent output stored in S3.

    Reads the CAPA output, applies a deterministic compliance rule engine to classify
    each sensor's actions as PASS or FAIL based on z-category severity, then uses
    AWS Bedrock (Meta Llama 3.1 8B) to generate concise summaries of preventive and
    corrective actions. Results are saved back to S3. Processes sensors concurrently
    using asyncio with a configurable semaphore.

    Args:
        capa_bucket: S3 bucket name containing the CAPA Agent output JSON
        capa_key: S3 object key of the CAPA Agent output JSON

    Returns:
        Dictionary containing:
        - status: "SUCCESS" or "FAILED"
        - compliance_output_s3_uri: S3 URI of the compliance output JSON (on success)
        - error: Error message string (on failure)
    """
    from agent_logic import run_compliance_engine, upload_output

    try:
        result = await run_compliance_engine(capa_bucket, capa_key)
        s3_uri = upload_output(result, capa_bucket)
        return {
            "status": "SUCCESS",
            "compliance_output_s3_uri": s3_uri
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
