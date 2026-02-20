#!/bin/bash
# =============================================================================
# build_and_push.sh
# Builds each agent's MCP server Docker image and pushes to Amazon ECR.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker running
#   - ECR repositories created (script creates them if missing)
#
# Usage:
#   chmod +x build_and_push.sh
#   AWS_ACCOUNT_ID=495688866359 AWS_PROFILE=agent-mesh-poc-495688866359 ./build_and_push.sh
# =============================================================================

set -e

# -------------------------------------------------
# CONFIGURATION — update these values
# -------------------------------------------------
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?ERROR: AWS_ACCOUNT_ID env var is required}"
REGION="us-west-2"
REPO_PREFIX="agent-mesh"

# Use AWS_PROFILE if set (e.g. agent-mesh-poc-495688866359)
if [ -n "${AWS_PROFILE}" ]; then
    export AWS_PROFILE
    echo "Using AWS profile: ${AWS_PROFILE}"
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# -------------------------------------------------
# Agent names and their mcp_server directories
# Format: "agent-name:relative/path/to/mcp_server"
# -------------------------------------------------
AGENTS=(
    "deviation-agent:../deviation_agent/mcp_server"
    "rca-agent:../rca_agent/mcp_server"
    "capa-agent:../capa_agent/mcp_server"
    "compliance-agent:../compliance_agent/mcp_server"
)

# -------------------------------------------------
# Login to ECR
# -------------------------------------------------
echo "Logging in to ECR..."
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# -------------------------------------------------
# Build and push each agent
# -------------------------------------------------
for ENTRY in "${AGENTS[@]}"; do
    AGENT_NAME="${ENTRY%%:*}"
    CONTEXT_DIR="${ENTRY##*:}"
    REPO_NAME="${REPO_PREFIX}/${AGENT_NAME}"
    IMAGE_URI="${ECR_REGISTRY}/${REPO_NAME}:latest"

    # Create ECR repository if it doesn't exist
    echo ""
    echo "Ensuring ECR repository exists: ${REPO_NAME}"
    aws ecr describe-repositories \
        --repository-names "${REPO_NAME}" \
        --region "${REGION}" > /dev/null 2>&1 \
    || aws ecr create-repository \
        --repository-name "${REPO_NAME}" \
        --region "${REGION}" \
        --image-scanning-configuration scanOnPush=true \
        > /dev/null

    echo "============================================"
    echo "Building: ${AGENT_NAME}"
    echo "Context:  ${CONTEXT_DIR}"
    echo "Image:    ${IMAGE_URI}"
    echo "============================================"

    docker build \
        --platform linux/arm64 \
        -t "${AGENT_NAME}:latest" \
        "${CONTEXT_DIR}"

    docker tag "${AGENT_NAME}:latest" "${IMAGE_URI}"

    echo "Pushing ${IMAGE_URI}..."
    docker push "${IMAGE_URI}"

    echo "Done: ${AGENT_NAME} → ${IMAGE_URI}"
done

echo ""
echo "============================================"
echo "All agents built and pushed successfully."
echo ""
echo "ECR Image URIs:"
for ENTRY in "${AGENTS[@]}"; do
    AGENT_NAME="${ENTRY%%:*}"
    echo "  ${AGENT_NAME}: ${ECR_REGISTRY}/${REPO_PREFIX}/${AGENT_NAME}:latest"
done
echo ""
echo "Next step: run create_agentcore_agents.py"
echo "============================================"
