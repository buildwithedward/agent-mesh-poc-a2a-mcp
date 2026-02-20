"""
a2a_orchestrator.py
===================
A2A (Agent-to-Agent) orchestrator for the SECOM Anomaly Analysis pipeline.

Uses:
- Official a2a-sdk (a2aproject/a2a-python) for AgentCard / AgentSkill / Task data models
- AgentCore dynamic discovery (list_agent_runtimes → get_agent_card)
- AWS Bedrock Claude to read AgentCards and plan the execution pipeline
- AgentCore invoke_agent_runtime to call each MCP tool
- Generator pattern so Streamlit can display live progress
"""

import boto3
import json
import uuid
from typing import Generator

from a2a.types import AgentCard, AgentCapabilities, AgentSkill, Task, TaskState, TaskStatus, Message, Role, Part, TextPart

from orchestrator.agentcore_mcp_client import AgentCoreMCPClient


# ─────────────────────────────────────────────
# Hardcoded context — S3 model location
# ─────────────────────────────────────────────
MODEL_BUCKET = "ag-agent-mesh"
MODEL_KEY = "Isolation_Forest_Model/isolation_forest_secom.pkl"
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


class AgentMeshOrchestrator:
    """
    A2A orchestrator that:
    1. Dynamically discovers all AgentCore Runtime agents
    2. Fetches their AgentCards
    3. Asks Bedrock Claude to plan the pipeline based on cards + task description
    4. Executes the pipeline step-by-step via MCP, streaming results
    """

    def __init__(self, region: str = "us-west-2"):
        self.region = region
        self.mcp_client = AgentCoreMCPClient(region=region)
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)

    # ──────────────────────────────────────────
    # Agent Discovery + Card Fetching
    # ──────────────────────────────────────────
    def _discover_and_fetch_cards(self) -> list[dict]:
        """
        Discover all READY AgentCore Runtime agents and fetch their AgentCards.
        Returns a list of enriched card dicts (with agentRuntimeArn + agentRuntimeName).
        """
        runtimes = self.mcp_client.discover_agents()
        enriched_cards = []

        for rt in runtimes:
            arn = rt["agentRuntimeArn"]
            name = rt["agentRuntimeName"]

            card = self.mcp_client.get_agent_card(
                agent_runtime_arn=arn,
                agent_name=name
            )

            # Ensure ARN and name are always present in the card
            card["agentRuntimeArn"] = arn
            card["agentRuntimeName"] = name
            enriched_cards.append(card)

        return enriched_cards

    def _cards_to_a2a_objects(self, raw_cards: list[dict]) -> list[AgentCard]:
        """Convert raw card dicts to official a2a-sdk AgentCard objects."""
        a2a_cards = []
        for c in raw_cards:
            skills = [
                AgentSkill(
                    id=s.get("id", s.get("name", "unknown")),
                    name=s.get("name", ""),
                    description=s.get("description", ""),
                    tags=s.get("tags", []),
                    input_modes=s.get("inputModes", s.get("input_modes", None)),
                    output_modes=s.get("outputModes", s.get("output_modes", None))
                )
                for s in c.get("skills", [])
            ]
            a2a_card = AgentCard(
                name=c.get("agentRuntimeName", c.get("name", "")),
                description=c.get("description", ""),
                url=c.get("agentRuntimeArn", c.get("url", "")),
                version=c.get("version", "1.0.0"),
                skills=skills,
                capabilities=AgentCapabilities(),
                default_input_modes=["application/json"],
                default_output_modes=["application/json"]
            )
            a2a_cards.append(a2a_card)
        return a2a_cards

    # ──────────────────────────────────────────
    # LLM Pipeline Planning
    # ──────────────────────────────────────────
    def _llm_plan_pipeline(self, raw_cards: list[dict], task_description: str) -> list[dict]:
        """
        Ask Bedrock Claude to read AgentCards and return an ordered execution pipeline.

        Args:
            raw_cards: Enriched agent card dicts
            task_description: Natural language description of the task

        Returns:
            List of steps: [{agent_name, agent_arn, tool_name, reason}]
        """
        # Build a concise summary of available agents for the LLM
        cards_summary = []
        for c in raw_cards:
            cards_summary.append({
                "agent_name": c.get("agentRuntimeName", c.get("name", "")),
                "agent_arn": c.get("agentRuntimeArn", c.get("url", "")),
                "description": c.get("description", ""),
                "skills": [
                    {
                        "tool_name": s.get("name", ""),
                        "description": s.get("description", ""),
                        "tags": s.get("tags", [])
                    }
                    for s in c.get("skills", [])
                ]
            })

        system_prompt = (
            "You are an intelligent pipeline orchestrator for a manufacturing anomaly analysis system. "
            "Given available agents and their skills, determine the optimal ordered pipeline to complete the task. "
            "Return ONLY a valid JSON array — no markdown, no explanation, no extra text.\n\n"
            "Each element must have exactly these fields:\n"
            '  "agent_name": agent name exactly as given\n'
            '  "agent_arn": agent ARN exactly as given\n'
            '  "tool_name": the specific tool/skill name to call\n'
            '  "reason": brief explanation of why this agent is at this position\n\n'
            "Order agents logically: anomaly detection → root cause analysis → "
            "corrective/preventive actions → compliance evaluation."
        )

        user_content = (
            f"Task: {task_description}\n\n"
            f"Available Agents:\n{json.dumps(cards_summary, indent=2)}\n\n"
            "Return the pipeline as a JSON array only."
        )

        response = self.bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
                "max_tokens": 1024,
                "temperature": 0
            }),
            contentType="application/json",
            accept="application/json"
        )

        body = json.loads(response["body"].read())
        text = body["content"][0]["text"].strip()

        # Strip markdown fences if present
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

        return json.loads(text)

    # ──────────────────────────────────────────
    # Argument Resolution
    # ──────────────────────────────────────────
    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Parse s3://bucket/key into (bucket, key)."""
        path = s3_uri.replace("s3://", "")
        bucket, _, key = path.partition("/")
        return bucket, key

    def _resolve_arguments(self, tool_name: str, context: dict) -> dict:
        """
        Map the current pipeline context to the correct MCP tool arguments.
        Context accumulates outputs from each previous step.
        """
        if tool_name == "run_deviation_detection":
            return {
                "dataset_bucket": context["dataset_bucket"],
                "dataset_key": context["dataset_key"]
            }

        if tool_name == "run_rca_analysis":
            deviation_uri = context.get("deviation_s3_uri")
            if not deviation_uri:
                raise ValueError(
                    f"Expected 'deviation_s3_uri' in context but it is missing. "
                    f"Context keys present: {list(context.keys())}. "
                    f"The deviation agent may have returned an unexpected result structure."
                )
            dev_bucket, dev_key = self._parse_s3_uri(deviation_uri)
            return {
                "dataset_bucket": context["dataset_bucket"],
                "dataset_key": context["dataset_key"],
                "model_bucket": MODEL_BUCKET,
                "model_key": MODEL_KEY,
                "deviation_bucket": dev_bucket,
                "deviation_key": dev_key
            }

        if tool_name == "run_capa_selection":
            rca_uri = context.get("rca_output_s3_uri")
            if not rca_uri:
                raise ValueError(
                    f"Expected 'rca_output_s3_uri' in context but it is missing. "
                    f"Context keys present: {list(context.keys())}."
                )
            rca_bucket, rca_key = self._parse_s3_uri(rca_uri)
            return {
                "rca_bucket": rca_bucket,
                "rca_key": rca_key
            }

        if tool_name == "run_compliance_evaluation":
            capa_uri = context.get("capa_output_s3_uri")
            if not capa_uri:
                raise ValueError(
                    f"Expected 'capa_output_s3_uri' in context but it is missing. "
                    f"Context keys present: {list(context.keys())}."
                )
            capa_bucket, capa_key = self._parse_s3_uri(capa_uri)
            return {
                "capa_bucket": capa_bucket,
                "capa_key": capa_key
            }

        # Unknown tool — pass full context as fallback
        return context

    def _update_context(self, context: dict, result: dict):
        """Accumulate S3 output URIs from each step into the shared context."""
        for key in ("deviation_s3_uri", "rca_output_s3_uri",
                    "capa_output_s3_uri", "compliance_output_s3_uri"):
            if key in result:
                context[key] = result[key]

    # ──────────────────────────────────────────
    # Main Pipeline Runner (Generator)
    # ──────────────────────────────────────────
    def run_pipeline(
        self,
        dataset_bucket: str,
        dataset_key: str
    ) -> Generator[dict, None, None]:
        """
        Execute the full agent pipeline for a SECOM dataset.
        Yields progress events for the Streamlit UI.

        Event types:
          {type: "discovery",       agents_found: int, agents: [...]}
          {type: "cards_fetched",   cards: [...]}
          {type: "pipeline_planned",steps: [...]}
          {type: "step_start",      step_num: int, total: int, agent_name, tool_name, reason, arguments}
          {type: "step_done",       agent_name, tool_name, result}
          {type: "step_error",      agent_name, tool_name, error}
          {type: "final",           result, compliance_s3_uri}
          {type: "error",           message}
        """
        # Create a2a Task to track the overall job
        task = Task(
            id=str(uuid.uuid4()),
            context_id=str(uuid.uuid4()),
            status=TaskStatus(state=TaskState.submitted),
            history=[]
        )

        # Shared pipeline context
        context = {
            "dataset_bucket": dataset_bucket,
            "dataset_key": dataset_key,
            "task_id": task.id
        }

        try:
            # ── Step 1: Discover agents ──────────────────
            raw_cards = self._discover_and_fetch_cards()
            yield {
                "type": "discovery",
                "agents_found": len(raw_cards),
                "agents": [c.get("agentRuntimeName", "") for c in raw_cards]
            }

            # ── Step 2: Build A2A AgentCard objects ──────
            a2a_cards = self._cards_to_a2a_objects(raw_cards)
            yield {
                "type": "cards_fetched",
                "cards": [
                    {
                        "name": card.name,
                        "description": card.description,
                        "skills": [s.name for s in (card.skills or [])]
                    }
                    for card in a2a_cards
                ]
            }

            # ── Step 3: LLM plans the pipeline ───────────
            task_description = (
                f"Analyze the SECOM manufacturing dataset at "
                f"s3://{dataset_bucket}/{dataset_key} for anomalies, "
                "determine root causes, select corrective/preventive actions, "
                "and evaluate compliance."
            )
            pipeline_steps = self._llm_plan_pipeline(raw_cards, task_description)
            yield {
                "type": "pipeline_planned",
                "steps": pipeline_steps
            }

            # ── Step 4: Execute pipeline ──────────────────
            total = len(pipeline_steps)
            for i, step in enumerate(pipeline_steps, start=1):
                agent_name = step["agent_name"]
                agent_arn = step["agent_arn"]
                tool_name = step["tool_name"]
                reason = step.get("reason", "")

                arguments = self._resolve_arguments(tool_name, context)

                yield {
                    "type": "step_start",
                    "step_num": i,
                    "total": total,
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    "reason": reason,
                    "arguments": arguments
                }

                try:
                    result = self.mcp_client.call_tool(
                        agent_runtime_arn=agent_arn,
                        tool_name=tool_name,
                        arguments=arguments
                    )
                    self._update_context(context, result)

                    # Validate expected output key was produced
                    _expected_keys = {
                        "run_deviation_detection": "deviation_s3_uri",
                        "run_rca_analysis": "rca_output_s3_uri",
                        "run_capa_selection": "capa_output_s3_uri",
                        "run_compliance_evaluation": "compliance_output_s3_uri",
                    }
                    expected_key = _expected_keys.get(tool_name)
                    if expected_key and expected_key not in context:
                        raise ValueError(
                            f"Tool '{tool_name}' completed but expected output key "
                            f"'{expected_key}' is missing from result. "
                            f"Actual result returned: {result}"
                        )

                    yield {
                        "type": "step_done",
                        "agent_name": agent_name,
                        "tool_name": tool_name,
                        "result": result
                    }

                except Exception as e:
                    yield {
                        "type": "step_error",
                        "agent_name": agent_name,
                        "tool_name": tool_name,
                        "error": str(e)
                    }
                    raise  # Halt pipeline on agent failure

            # ── Step 5: Final result ──────────────────────
            compliance_uri = context.get("compliance_output_s3_uri", "")
            final_result = {
                "task_id": task.id,
                "dataset_s3_uri": f"s3://{dataset_bucket}/{dataset_key}",
                "deviation_s3_uri": context.get("deviation_s3_uri", ""),
                "rca_output_s3_uri": context.get("rca_output_s3_uri", ""),
                "capa_output_s3_uri": context.get("capa_output_s3_uri", ""),
                "compliance_output_s3_uri": compliance_uri
            }

            yield {
                "type": "final",
                "result": final_result,
                "compliance_s3_uri": compliance_uri
            }

        except Exception as e:
            yield {
                "type": "error",
                "message": str(e)
            }
