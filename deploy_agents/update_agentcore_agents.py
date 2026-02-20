"""
update_agentcore_agents.py
==========================
Updates existing AWS Bedrock AgentCore Runtime agents to pull the latest ECR image.
Run this AFTER pushing new Docker images with build_and_push.sh.

Usage:
    AWS_ACCOUNT_ID=495688866359 AGENTCORE_ROLE_ARN=arn:aws:iam::... python update_agentcore_agents.py

Environment Variables:
    AWS_ACCOUNT_ID      : Your AWS account ID
    AGENTCORE_ROLE_ARN  : (optional) IAM role ARN override for all AgentCore Runtimes.
                          If omitted, each runtime's existing roleArn is fetched and reused.
    QDRANT_URL          : (optional) Qdrant URL for CAPA agent
    QDRANT_API_KEY      : (optional) Qdrant API key for CAPA agent
"""

import boto3
import os
import json

REGION = "us-west-2"
AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "495688866359")
AGENTCORE_ROLE_ARN = os.environ.get("AGENTCORE_ROLE_ARN", "")

ECR_REGISTRY = f"{AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
REPO_PREFIX = "agent-mesh"

# Existing runtime ARNs (from agentcore_arns.json / prior deployment)
AGENTS = [
    {
        "name": "deviation_agent",
        "arn": "arn:aws:bedrock-agentcore:us-west-2:495688866359:runtime/deviation_agent-sYVlkgCOUk",
        "ecr_image": f"{ECR_REGISTRY}/{REPO_PREFIX}/deviation-agent:latest",
        "env_vars": {}
    },
    {
        "name": "rca_agent",
        "arn": "arn:aws:bedrock-agentcore:us-west-2:495688866359:runtime/rca_agent-ch55s730El",
        "ecr_image": f"{ECR_REGISTRY}/{REPO_PREFIX}/rca-agent:latest",
        "env_vars": {}
    },
    {
        "name": "capa_agent",
        "arn": "arn:aws:bedrock-agentcore:us-west-2:495688866359:runtime/capa_agent-E4QA65Dyf2",
        "ecr_image": f"{ECR_REGISTRY}/{REPO_PREFIX}/capa-agent:latest",
        "env_vars": {
            "VDB_URL": os.environ.get("QDRANT_URL", ""),
            "VDB_API": os.environ.get("QDRANT_API_KEY", "")
        }
    },
    {
        "name": "compliance_agent",
        "arn": "arn:aws:bedrock-agentcore:us-west-2:495688866359:runtime/compliance_agent-JbsQfp21dA",
        "ecr_image": f"{ECR_REGISTRY}/{REPO_PREFIX}/compliance-agent:latest",
        "env_vars": {}
    }
]


def _resolve_role_arn(client, runtime_id: str) -> str:
    """Resolve role ARN for update: explicit env override, else runtime's current roleArn."""
    if AGENTCORE_ROLE_ARN:
        return AGENTCORE_ROLE_ARN

    resp = client.get_agent_runtime(agentRuntimeId=runtime_id)
    role_arn = resp.get("roleArn", "")
    if not role_arn:
        raise ValueError(
            f"No roleArn found for runtime '{runtime_id}'. "
            "Set AGENTCORE_ROLE_ARN explicitly."
        )
    return role_arn


def update_agentcore_agent(client, agent: dict) -> dict:
    """Update a single AgentCore Runtime agent with the latest ECR image."""
    print(f"\n{'=' * 60}")
    print(f"Updating AgentCore Runtime agent: {agent['name']}")
    print(f"ARN:   {agent['arn']}")
    print(f"Image: {agent['ecr_image']}")

    # Extract the runtime ID from the ARN (last segment after /)
    runtime_id = agent["arn"].split("/")[-1]
    role_arn = _resolve_role_arn(client, runtime_id)

    env_vars = {k: v for k, v in agent["env_vars"].items() if v}

    update_kwargs = {
        "agentRuntimeId": runtime_id,
        "roleArn": role_arn,
        "agentRuntimeArtifact": {
            "containerConfiguration": {
                "containerUri": agent["ecr_image"]
            }
        },
        "networkConfiguration": {
            "networkMode": "PUBLIC"
        },
        "protocolConfiguration": {
            "serverProtocol": "MCP"
        }
    }

    if env_vars:
        update_kwargs["environmentVariables"] = env_vars

    response = client.update_agent_runtime(**update_kwargs)

    status = response.get("agentRuntimeStatus", response.get("status", "UNKNOWN"))
    print(f"  Status: {status}")
    print(f"  Role:   {role_arn}")

    return {"name": agent["name"], "arn": agent["arn"], "status": status}


def wait_for_ready(client, agents: list, timeout_seconds: int = 300):
    """Poll until all agents reach READY status."""
    import time
    print(f"\nWaiting for all agents to become READY (timeout={timeout_seconds}s)...")
    deadline = time.time() + timeout_seconds
    pending = {a["name"]: a["arn"].split("/")[-1] for a in agents}

    while pending and time.time() < deadline:
        time.sleep(15)
        for name in list(pending.keys()):
            runtime_id = pending[name]
            resp = client.get_agent_runtime(agentRuntimeId=runtime_id)
            status = resp.get("status", "UNKNOWN")
            print(f"  {name}: {status}")
            if status == "READY":
                del pending[name]
            elif status in ("FAILED", "DELETING"):
                print(f"  ERROR: {name} entered terminal state: {status}")
                del pending[name]

    if pending:
        print(f"\nTimeout reached. Still not READY: {list(pending.keys())}")
    else:
        print("\nAll agents are READY!")


def main():
    client = boto3.client("bedrock-agentcore-control", region_name=REGION)
    results = []

    for agent in AGENTS:
        try:
            result = update_agentcore_agent(client, agent)
            results.append(result)
        except Exception as e:
            print(f"  ERROR updating '{agent['name']}': {e}")
            results.append({"name": agent["name"], "status": "FAILED", "error": str(e)})

    print(f"\n{'=' * 60}")
    print("UPDATE SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['name']}: {r.get('status', 'UNKNOWN')}")
        if r.get("error"):
            print(f"    Error: {r['error']}")

    # Wait for READY status
    wait_for_ready(client, AGENTS)


if __name__ == "__main__":
    main()
