"""
create_agentcore_agents.py
==========================
Creates (or updates) AWS Bedrock AgentCore Runtime agents for each MCP-wrapped agent.
Each agent is deployed as a standalone MCP server container in AgentCore Runtime.

Prerequisites:
    pip install boto3

Usage:
    AWS_ACCOUNT_ID=123456789012 AGENTCORE_ROLE_ARN=arn:aws:iam::... python create_agentcore_agents.py

Environment Variables:
    AWS_ACCOUNT_ID      : Your AWS account I  : IAM role ARN for AgentCore Runtime (must have S3, Bedrock, ECR access)
    QDRANT_URL          : (optional) Qdrant URL for CAPA agent — set here or in container env
    QDRANT_API_KEY      : (optional) Qdrant API key for CAPA agent — set here or in container env
"""

import boto3
import os
import json

# =============================================================================
# CONFIGURATION
# =============================================================================
REGION = "us-west-2"
AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "")
AGENTCORE_ROLE_ARN = os.environ.get("AGENTCORE_ROLE_ARN", "")

if not AWS_ACCOUNT_ID:
    raise ValueError("AWS_ACCOUNT_ID environment variable is required")
if not AGENTCORE_ROLE_ARN:
    raise ValueError("AGENTCORE_ROLE_ARN environment variable is required")

ECR_REGISTRY = f"{AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
REPO_PREFIX = "agent-mesh"

# =============================================================================
# AGENT DEFINITIONS
# Each entry defines one AgentCore Runtime agent.
# env_vars: environment variables injected into the running container.
# =============================================================================
AGENTS = [
    {
        "name": "deviation_agent",
        "description": "Isolation Forest anomaly detection MCP server",
        "ecr_image": f"{ECR_REGISTRY}/{REPO_PREFIX}/deviation-agent:latest",
        "env_vars": {}
    },
    {
        "name": "rca_agent",
        "description": "Root Cause Analysis with SHAP explainability MCP server",
        "ecr_image": f"{ECR_REGISTRY}/{REPO_PREFIX}/rca-agent:latest",
        "env_vars": {}
    },
    {
        "name": "capa_agent",
        "description": "Corrective and Preventive Actions selection from Qdrant MCP server",
        "ecr_image": f"{ECR_REGISTRY}/{REPO_PREFIX}/capa-agent:latest",
        "env_vars": {
            "VDB_URL": os.environ.get("QDRANT_URL", ""),
            "VDB_API": os.environ.get("QDRANT_API_KEY", "")
        }
    },
    {
        "name": "compliance_agent",
        "description": "Compliance evaluation with AWS Bedrock summarization MCP server",
        "ecr_image": f"{ECR_REGISTRY}/{REPO_PREFIX}/compliance-agent:latest",
        "env_vars": {}
    }
]

# =============================================================================
# CREATE AGENTCORE RUNTIME AGENTS
# =============================================================================
def create_agentcore_agent(client, agent: dict) -> dict:
    """Create a single AgentCore Runtime agent and return its ARN and endpoint."""

    print(f"\n{'=' * 60}")
    print(f"Creating AgentCore Runtime agent: {agent['name']}")
    print(f"Image: {agent['ecr_image']}")

    # Build environment variables dict
    env_vars = {
        k: v
        for k, v in agent["env_vars"].items()
        if v  # skip empty values
    }

    create_kwargs = {
        "agentRuntimeName": agent["name"],
        "description": agent["description"],
        "agentRuntimeArtifact": {
            "containerConfiguration": {
                "containerUri": agent["ecr_image"]
            }
        },
        "roleArn": AGENTCORE_ROLE_ARN,
        "protocolConfiguration": {
            "serverProtocol": "MCP"
        },
        "networkConfiguration": {
            "networkMode": "PUBLIC"
        }
    }

    if env_vars:
        create_kwargs["environmentVariables"] = env_vars

    response = client.create_agent_runtime(**create_kwargs)

    arn = response.get("agentRuntimeArn", "")
    endpoint = response.get("agentRuntimeEndpoint", "")

    print(f"  ARN:      {arn}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Status:   {response.get('status', 'UNKNOWN')}")

    return {
        "name": agent["name"],
        "agentRuntimeArn": arn,
        "agentRuntimeEndpoint": endpoint,
        "status": response.get("status", "UNKNOWN")
    }


def main():
    client = boto3.client("bedrock-agentcore-control", region_name=REGION)

    results = []

    for agent in AGENTS:
        try:
            result = create_agentcore_agent(client, agent)
            results.append(result)
        except client.exceptions.ConflictException:
            print(f"  Agent '{agent['name']}' already exists — skipping.")
            results.append({"name": agent["name"], "status": "ALREADY_EXISTS"})
        except Exception as e:
            print(f"  ERROR creating '{agent['name']}': {e}")
            results.append({"name": agent["name"], "status": "FAILED", "error": str(e)})

    # -------------------------------------------------
    # Print summary
    # -------------------------------------------------
    print(f"\n{'=' * 60}")
    print("DEPLOYMENT SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"\n  Agent : {r['name']}")
        print(f"  Status: {r.get('status', 'UNKNOWN')}")
        if r.get("agentRuntimeArn"):
            print(f"  ARN   : {r['agentRuntimeArn']}")
        if r.get("agentRuntimeEndpoint"):
            print(f"  URL   : {r['agentRuntimeEndpoint']}")
        if r.get("error"):
            print(f"  Error : {r['error']}")

    # -------------------------------------------------
    # Save ARNs to a JSON file for future use (e.g., A2A config)
    # -------------------------------------------------
    output_file = "agentcore_arns.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nARNs saved to: {output_file}")


if __name__ == "__main__":
    main()
