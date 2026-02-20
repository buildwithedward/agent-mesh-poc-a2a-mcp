"""
agentcore_mcp_client.py
=======================
Handles all communication with AWS Bedrock AgentCore Runtime agents via MCP protocol.

- Dynamically discovers all deployed READY agents (no hardcoded ARNs)
- Fetches AgentCards from AgentCore API (falls back to local JSON if unavailable)
- Calls agent MCP tools via invoke_agent_runtime using JSON-RPC 2.0 payloads
"""

import boto3
import json
import os
import uuid
from typing import Optional


# ─────────────────────────────────────────────
# Local fallback card directory
# ─────────────────────────────────────────────
_CARDS_DIR = os.path.join(os.path.dirname(__file__), "agent_cards")


class AgentCoreMCPClient:
    """
    Client for discovering and invoking AgentCore Runtime MCP agents.
    All agent discovery is dynamic — no hardcoded ARNs.
    """

    def __init__(self, region: str = "us-west-2"):
        self.region = region
        # Control plane: list / describe agents
        self.control = boto3.client("bedrock-agentcore-control", region_name=region)
        # Data plane: get agent card + invoke
        self.runtime = boto3.client("bedrock-agentcore", region_name=region)

    # ──────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────
    def discover_agents(self) -> list[dict]:
        """
        Discover all READY AgentCore Runtime agents dynamically.

        Returns:
            List of dicts: [{agentRuntimeArn, agentRuntimeName, agentRuntimeId, status}]
        """
        agents = []
        paginator_kwargs = {}

        while True:
            response = self.control.list_agent_runtimes(**paginator_kwargs)
            for runtime in response.get("agentRuntimes", []):
                if runtime.get("status") == "READY":
                    agents.append({
                        "agentRuntimeArn": runtime["agentRuntimeArn"],
                        "agentRuntimeName": runtime["agentRuntimeName"],
                        "agentRuntimeId": runtime["agentRuntimeId"],
                        "status": runtime["status"]
                    })
            next_token = response.get("nextToken")
            if not next_token:
                break
            paginator_kwargs["nextToken"] = next_token

        return agents

    # ──────────────────────────────────────────
    # Agent Card
    # ──────────────────────────────────────────
    def get_agent_card(self, agent_runtime_arn: str, agent_name: str = None) -> dict:
        """
        Fetch the AgentCard for a given AgentCore Runtime agent.
        Falls back to local JSON card if the API returns no usable card.

        Args:
            agent_runtime_arn: Full ARN of the AgentCore Runtime agent
            agent_name: Agent name (used for local fallback lookup)

        Returns:
            AgentCard dict with name, description, skills, url
        """
        try:
            response = self.runtime.get_agent_card(agentRuntimeArn=agent_runtime_arn)
            card = response.get("agentCard") or response.get("card") or {}

            # Enrich with ARN as url if not present
            if isinstance(card, dict) and card.get("skills"):
                card.setdefault("url", agent_runtime_arn)
                return card
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                f"[AgentCard] get_agent_card not supported for MCP agent {agent_runtime_arn}, "
                f"falling back to local card. ({e})"
            )

        # Fall back to local JSON card
        return self._load_local_card(agent_name or agent_runtime_arn)

    def _load_local_card(self, agent_name: str) -> dict:
        """Load a local fallback AgentCard JSON by agent name."""
        card_path = os.path.join(_CARDS_DIR, f"{agent_name}_card.json")
        if os.path.exists(card_path):
            with open(card_path) as f:
                card = json.load(f)
            return card

        # Minimal card if nothing found
        return {
            "name": agent_name,
            "description": f"Agent: {agent_name}",
            "skills": []
        }

    # ──────────────────────────────────────────
    # MCP Tool Invocation
    # ──────────────────────────────────────────
    def call_tool(
        self,
        agent_runtime_arn: str,
        tool_name: str,
        arguments: dict,
        session_id: Optional[str] = None
    ) -> dict:
        """
        Invoke an MCP tool on a deployed AgentCore Runtime agent.

        Performs the full MCP handshake:
          1. initialize  — establishes MCP session, returns mcpSessionId
          2. notifications/initialized — confirms session
          3. tools/call  — invokes the tool

        Args:
            agent_runtime_arn: Full ARN of the target agent
            tool_name: MCP tool name to invoke (e.g. "run_deviation_detection")
            arguments: Tool arguments dict
            session_id: Optional runtime session ID (auto-generated if not provided)

        Returns:
            Parsed tool result dict
        """
        runtime_session_id = session_id or str(uuid.uuid4())
        # MCP streamable-http requires both JSON and SSE in Accept header
        accept = "application/json, text/event-stream"

        # ── 1. Initialize MCP session ────────────────────────────────────
        init_resp = self.runtime.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            payload=json.dumps({
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "a2a-orchestrator", "version": "1.0.0"}
                }
            }).encode("utf-8"),
            contentType="application/json",
            accept=accept,
            runtimeSessionId=runtime_session_id
        )
        mcp_session_id = init_resp.get("mcpSessionId") or ""
        # Consume the init response body to allow the connection to complete
        init_body = init_resp.get("response")
        if init_body:
            init_raw = init_body.read()
            print(f"  [MCP Init Response]: {init_raw[:300]}")

        # ── 2. Send initialized notification ────────────────────────────
        notif_kwargs = dict(
            agentRuntimeArn=agent_runtime_arn,
            payload=json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }).encode("utf-8"),
            contentType="application/json",
            accept=accept,
            runtimeSessionId=runtime_session_id
        )
        if mcp_session_id:
            notif_kwargs["mcpSessionId"] = mcp_session_id
        notif_resp = self.runtime.invoke_agent_runtime(**notif_kwargs)
        # Consume the notification response body
        notif_body = notif_resp.get("response")
        if notif_body:
            notif_raw = notif_body.read()
            print(f"  [MCP Notif Response]: {notif_raw[:200]}")

        # ── 3. Call the tool ─────────────────────────────────────────────
        tool_kwargs = dict(
            agentRuntimeArn=agent_runtime_arn,
            payload=json.dumps({
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }).encode("utf-8"),
            contentType="application/json",
            accept=accept,
            runtimeSessionId=runtime_session_id
        )
        if mcp_session_id:
            tool_kwargs["mcpSessionId"] = mcp_session_id
        tool_resp = self.runtime.invoke_agent_runtime(**tool_kwargs)

        # NOTE: response body is in key "response", not "payload"
        raw = tool_resp["response"].read()
        return self._parse_mcp_response(raw)

    def _parse_mcp_response(self, raw: bytes) -> dict:
        """
        Parse the MCP JSON-RPC response from invoke_agent_runtime.

        Handles both plain JSON and SSE (text/event-stream) formats.

        FastMCP plain JSON:
          {"jsonrpc":"2.0","id":"...","result":{"content":[{"type":"text","text":"..."}]}}

        FastMCP SSE:
          data: {"jsonrpc":"2.0","id":"...","result":{...}}
        """
        text = raw.decode("utf-8", errors="replace").strip()
        print(f"  [MCP Raw Response] ({len(raw)} bytes): {text[:500]}")

        # Unwrap SSE format → collect all non-DONE data lines, use last one
        if text.startswith("data:") or "\ndata:" in text:
            json_str = None
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    candidate = line[len("data:"):].strip()
                    if candidate and candidate != "[DONE]":
                        json_str = candidate
            if json_str:
                text = json_str
                print(f"  [MCP SSE Unwrapped]: {text[:300]}")

        try:
            envelope = json.loads(text)
        except Exception:
            return {"status": "FAILED", "error": f"Non-JSON response: {text[:300]}"}

        if "error" in envelope:
            err = envelope["error"]
            return {
                "status": "FAILED",
                "error": err.get("message", str(err)) if isinstance(err, dict) else str(err)
            }

        result = envelope.get("result", {})
        print(f"  [MCP Result envelope]: {str(result)[:300]}")

        # MCP content array → extract first text part and parse as JSON
        content = result.get("content", [])
        if content and isinstance(content, list):
            first = content[0]
            part_text = first.get("text", "") if isinstance(first, dict) else str(first)
            print(f"  [MCP Content text]: {part_text[:300]}")
            try:
                parsed = json.loads(part_text)
                print(f"  [MCP Parsed result]: {parsed}")
                return parsed
            except json.JSONDecodeError:
                # part_text might already be a plain string result — return as-is
                return {"status": "SUCCESS", "raw_output": part_text}

        # No content array — return result dict directly if it has useful keys
        if result:
            return result

        return {"status": "SUCCESS", "raw_response": text[:500]}
