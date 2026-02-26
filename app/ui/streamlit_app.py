"""
streamlit_app.py
================
SECOM Anomaly Analysis — Agent Mesh UI

Allows users to upload a SECOM dataset (CSV), which is stored in S3 and
processed through the A2A orchestrated agent pipeline. Live intermediate
steps and final compliance results are displayed in the UI.

Tabs:
  1. Pipeline   — Upload CSV + run the agent pipeline (existing flow)
  2. Agent Cards — View details of all discovered agents and their skills
  3. I/O Summary — Per-agent input/output summary with durations
  4. Agent Flow  — Interactive pipeline graph with clickable nodes
"""

import sys
import os
import json
import time
import base64
import glob as globmod
import boto3
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from dotenv import load_dotenv

# ── Make orchestrator importable from app/ root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.a2a_orchestrator import AgentMeshOrchestrator

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
S3_BUCKET = os.environ.get("S3_BUCKET", "ag-agent-mesh")
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
INPUT_PREFIX = "input"
_CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "orchestrator", "agent_cards")
_LOGO_DIR = os.path.join(os.path.dirname(__file__), "logo")


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = []
if "agent_cards_data" not in st.session_state:
    st.session_state.agent_cards_data = []
if "final_result" not in st.session_state:
    st.session_state.final_result = None
if "pipeline_plan" not in st.session_state:
    st.session_state.pipeline_plan = []
if "discovery_info" not in st.session_state:
    st.session_state.discovery_info = None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def upload_to_s3(file_bytes: bytes, filename: str) -> str:
    """Upload dataset bytes to S3 and return the S3 key."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    key = f"{INPUT_PREFIX}/{timestamp}_{filename}"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType="text/csv"
    )
    return key


def fetch_s3_json(s3_uri: str) -> dict | list | None:
    """Fetch and parse a JSON file from S3 given an s3:// URI."""
    try:
        path = s3_uri.replace("s3://", "")
        bucket, _, key = path.partition("/")
        s3 = boto3.client("s3", region_name=AWS_REGION)
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read())
    except Exception as e:
        return {"error": str(e)}


def load_agent_cards() -> list[dict]:
    """Load all agent card JSON files from the local agent_cards directory."""
    cards = []
    card_files = sorted(globmod.glob(os.path.join(_CARDS_DIR, "*_card.json")))
    for fp in card_files:
        with open(fp) as f:
            cards.append(json.load(f))
    return cards


def get_logo_base64(filename: str) -> str | None:
    """Read a logo file and return its base64-encoded data URI."""
    filepath = os.path.join(_LOGO_DIR, filename)
    if not os.path.exists(filepath):
        return None
    ext = os.path.splitext(filename)[1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        ext.lstrip("."), "image/png"
    )
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{encoded}"


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SECOM Anomaly Analysis",
    page_icon="🔬",
    layout="wide"
)

# ─────────────────────────────────────────────
# Logos + Title Header
# ─────────────────────────────────────────────
_logo_left = get_logo_base64("Agilisium.png")
_logo_right = get_logo_base64("phlow.jpg")

if _logo_left and _logo_right:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; padding:0 0 10px 0;">
            <img src="{_logo_left}" style="height:60px;" alt="Agilisium Logo">
            <h2 style="margin:0; text-align:center;">🔬 SECOM Anomaly Analysis — Agent Mesh</h2>
            <img src="{_logo_right}" style="height:60px;" alt="Phlow Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )
elif _logo_left:
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:16px; padding-bottom:10px;">'
        f'<img src="{_logo_left}" style="height:60px;" alt="Logo">'
        f'<h2 style="margin:0;">🔬 SECOM Anomaly Analysis — Agent Mesh</h2></div>',
        unsafe_allow_html=True,
    )
else:
    st.title("🔬 SECOM Anomaly Analysis — Agent Mesh")

st.caption(
    "Upload a SECOM manufacturing dataset. The A2A orchestrator will dynamically "
    "discover agents, plan the pipeline via LLM, and execute each step via MCP."
)
st.divider()

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_pipeline, tab_cards, tab_io, tab_flow = st.tabs([
    "Pipeline", "Agent Cards", "I/O Summary", "Agent Flow"
])


# ═══════════════════════════════════════════════
# TAB 1: Pipeline (existing flow)
# ═══════════════════════════════════════════════
with tab_pipeline:
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload SECOM Dataset (CSV)",
            type=["csv"],
            help="Upload the SECOM manufacturing sensor dataset in CSV format."
        )

    with col2:
        if uploaded_file:
            st.success(f"File ready: **{uploaded_file.name}**")
            st.metric("File size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")

    st.divider()

    if uploaded_file:
        if st.button("▶ Run Analysis", type="primary", use_container_width=True):

            # Reset session state for new run
            st.session_state.pipeline_steps = []
            st.session_state.agent_cards_data = []
            st.session_state.final_result = None
            st.session_state.pipeline_plan = []
            st.session_state.discovery_info = None

            file_bytes = uploaded_file.getvalue()

            # ── Upload to S3 ──────────────────────────
            with st.spinner("Uploading dataset to S3..."):
                dataset_key = upload_to_s3(file_bytes, uploaded_file.name)
                dataset_s3_uri = f"s3://{S3_BUCKET}/{dataset_key}"

            st.success(f"Dataset uploaded → `{dataset_s3_uri}`")
            st.divider()

            # ── Pipeline Execution ────────────────────
            orchestrator = AgentMeshOrchestrator(region=AWS_REGION)
            final_result = None
            pipeline_steps_display = []
            step_timers = {}  # agent_name → start_time

            with st.status("🤖 Running Agent Pipeline...", expanded=True) as pipeline_status:

                for event in orchestrator.run_pipeline(S3_BUCKET, dataset_key):

                    etype = event.get("type")

                    # ── Agent Discovery ──────────────────
                    if etype == "discovery":
                        n = event["agents_found"]
                        names = ", ".join(event["agents"])
                        st.write(f"🔍 **Agent Discovery** — {n} agents found: `{names}`")
                        st.session_state.discovery_info = {
                            "agents_found": n,
                            "agents": event["agents"]
                        }

                    # ── Cards Fetched ────────────────────
                    elif etype == "cards_fetched":
                        st.session_state.agent_cards_data = event["cards"]
                        with st.expander("📋 AgentCards fetched", expanded=False):
                            for card in event["cards"]:
                                st.markdown(f"**{card['name']}**")
                                st.caption(card["description"])
                                if card["skills"]:
                                    st.markdown("Skills: " + ", ".join(f"`{s}`" for s in card["skills"]))

                    # ── Pipeline Plan ────────────────────
                    elif etype == "pipeline_planned":
                        steps = event["steps"]
                        st.session_state.pipeline_plan = steps
                        with st.expander(f"🧠 LLM planned {len(steps)}-step pipeline", expanded=True):
                            for i, s in enumerate(steps, 1):
                                st.markdown(
                                    f"**{i}. {s['agent_name']}** → `{s['tool_name']}`  \n"
                                    f"_{s.get('reason', '')}_"
                                )

                    # ── Step Start ───────────────────────
                    elif etype == "step_start":
                        num = event["step_num"]
                        total = event["total"]
                        agent = event["agent_name"]
                        tool = event["tool_name"]
                        pipeline_status.update(
                            label=f"⏳ Step {num}/{total}: calling **{agent}**..."
                        )
                        step_timers[agent] = time.time()
                        pipeline_steps_display.append({
                            "num": num,
                            "agent": agent,
                            "tool": tool,
                            "reason": event.get("reason", ""),
                            "arguments": event.get("arguments", {})
                        })

                    # ── Step Done ────────────────────────
                    elif etype == "step_done":
                        agent = event["agent_name"]
                        result = event["result"]
                        duration = time.time() - step_timers.get(agent, time.time())
                        step_info = next(
                            (s for s in pipeline_steps_display if s["agent"] == agent), {}
                        )

                        # Store in session state for I/O Summary and Agent Flow tabs
                        st.session_state.pipeline_steps.append({
                            "num": step_info.get("num", ""),
                            "agent": agent,
                            "tool": event["tool_name"],
                            "reason": step_info.get("reason", ""),
                            "arguments": step_info.get("arguments", {}),
                            "result": result,
                            "duration": round(duration, 2),
                            "status": "SUCCESS"
                        })

                        with st.expander(
                            f"✅ {step_info.get('num', '')}.  **{agent}** — completed ({duration:.1f}s)",
                            expanded=False
                        ):
                            st.markdown(f"**Tool:** `{event['tool_name']}`")
                            st.markdown(f"*{step_info.get('reason', '')}*")
                            col_in, col_out = st.columns(2)
                            with col_in:
                                st.markdown("**Inputs**")
                                st.json(step_info.get("arguments", {}))
                            with col_out:
                                st.markdown("**Output**")
                                st.json(result)

                    # ── Step Error ───────────────────────
                    elif etype == "step_error":
                        agent = event["agent_name"]
                        duration = time.time() - step_timers.get(agent, time.time())
                        step_info = next(
                            (s for s in pipeline_steps_display if s["agent"] == agent), {}
                        )
                        st.session_state.pipeline_steps.append({
                            "num": step_info.get("num", ""),
                            "agent": agent,
                            "tool": event["tool_name"],
                            "reason": step_info.get("reason", ""),
                            "arguments": step_info.get("arguments", {}),
                            "result": {"error": event["error"]},
                            "duration": round(duration, 2),
                            "status": "FAILED"
                        })
                        st.error(
                            f"❌ **{agent}** failed: {event['error']}"
                        )

                    # ── Final ────────────────────────────
                    elif etype == "final":
                        final_result = event["result"]
                        st.session_state.final_result = final_result
                        pipeline_status.update(
                            label="✅ Pipeline Complete!",
                            state="complete",
                            expanded=False
                        )

                    # ── Pipeline Error ───────────────────
                    elif etype == "error":
                        st.error(f"❌ Pipeline error: {event['message']}")
                        pipeline_status.update(
                            label="❌ Pipeline failed",
                            state="error",
                            expanded=True
                        )

            # ── Final Results ─────────────────────────
            if final_result:
                st.divider()
                st.subheader("📊 Pipeline Summary")

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Deviation", "✅ Done")
                r2.metric("RCA", "✅ Done")
                r3.metric("CAPA", "✅ Done")
                r4.metric("Compliance", "✅ Done")

                st.subheader("🗂 Output S3 URIs")
                for label, key in [
                    ("Dataset Input", "dataset_s3_uri"),
                    ("Deviation Output", "deviation_s3_uri"),
                    ("RCA Output", "rca_output_s3_uri"),
                    ("CAPA Output", "capa_output_s3_uri"),
                    ("Compliance Output", "compliance_output_s3_uri"),
                ]:
                    uri = final_result.get(key, "")
                    if uri:
                        st.markdown(f"**{label}:** `{uri}`")

                # ── Compliance Result Detail ──────────
                compliance_uri = final_result.get("compliance_output_s3_uri", "")
                if compliance_uri:
                    st.subheader("📋 Compliance Results")
                    with st.spinner("Fetching compliance output from S3..."):
                        compliance_data = fetch_s3_json(compliance_uri)

                    if compliance_data:
                        st.json(compliance_data)

                        st.download_button(
                            label="⬇ Download Compliance Report (JSON)",
                            data=json.dumps(compliance_data, indent=2),
                            file_name=f"compliance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            type="primary"
                        )

    else:
        st.info("👆 Upload a SECOM CSV dataset to get started.")


# ═══════════════════════════════════════════════
# TAB 2: Agent Cards
# ═══════════════════════════════════════════════
with tab_cards:
    st.subheader("📋 Agent Cards")
    st.caption("Details of all agents deployed in AWS Bedrock AgentCore Runtime.")
    st.divider()

    local_cards = load_agent_cards()

    if not local_cards:
        st.warning("No agent card files found in `app/orchestrator/agent_cards/`.")
    else:
        for card in local_cards:
            card_name = card.get("name", "Unknown Agent")
            card_desc = card.get("description", "No description")
            card_version = card.get("version", "—")
            card_url = card.get("url", "—")
            skills = card.get("skills", [])

            with st.expander(f"**{card_name}**  (v{card_version})", expanded=True):
                st.markdown(f"**Description:** {card_desc}")
                st.markdown(f"**ARN:** `{card_url}`")
                st.markdown(f"**Version:** {card_version}")

                if skills:
                    st.markdown("---")
                    st.markdown("**Skills:**")
                    for skill in skills:
                        skill_name = skill.get("name", "unknown")
                        skill_desc = skill.get("description", "")
                        skill_tags = skill.get("tags", [])
                        st.markdown(f"- **`{skill_name}`** — {skill_desc}")
                        if skill_tags:
                            st.markdown(
                                "  Tags: " + " ".join(f"`{t}`" for t in skill_tags)
                            )
                else:
                    st.info("No skills defined for this agent.")


# ═══════════════════════════════════════════════
# TAB 3: I/O Summary
# ═══════════════════════════════════════════════
with tab_io:
    st.subheader("📊 I/O Summary")
    st.caption("Input/output details and duration for each agent step in the pipeline.")
    st.divider()

    steps = st.session_state.pipeline_steps

    if not steps:
        st.info("Run the pipeline first (in the Pipeline tab) to see I/O summaries here.")
    else:
        # Summary metrics row
        total_duration = sum(s["duration"] for s in steps)
        success_count = sum(1 for s in steps if s["status"] == "SUCCESS")
        fail_count = sum(1 for s in steps if s["status"] == "FAILED")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Steps", len(steps))
        m2.metric("Succeeded", success_count)
        m3.metric("Failed", fail_count)
        m4.metric("Total Duration", f"{total_duration:.1f}s")

        st.divider()

        # Per-step details
        for step in steps:
            status_icon = "✅" if step["status"] == "SUCCESS" else "❌"
            header = (
                f"{status_icon} Step {step['num']}: **{step['agent']}** "
                f"→ `{step['tool']}` ({step['duration']}s)"
            )

            with st.expander(header, expanded=False):
                if step.get("reason"):
                    st.markdown(f"*{step['reason']}*")

                st.markdown(f"**Duration:** {step['duration']}s")
                st.markdown(f"**Status:** {step['status']}")

                col_in, col_out = st.columns(2)
                with col_in:
                    st.markdown("**Input Arguments**")
                    st.json(step["arguments"])
                with col_out:
                    st.markdown("**Output Result**")
                    st.json(step["result"])


# ═══════════════════════════════════════════════
# TAB 4: Agent Flow (Interactive Graph)
# ═══════════════════════════════════════════════
with tab_flow:
    st.subheader("🔗 Agent Flow")
    st.caption("Interactive pipeline graph — click an agent node to view its I/O details.")
    st.divider()

    steps = st.session_state.pipeline_steps

    if not steps:
        st.info("Run the pipeline first (in the Pipeline tab) to see the agent flow graph here.")
    else:
        # Build node data for the graph
        nodes_html_parts = []
        for i, step in enumerate(steps):
            color = "#28a745" if step["status"] == "SUCCESS" else "#dc3545"
            border_color = "#1e7e34" if step["status"] == "SUCCESS" else "#bd2130"
            status_icon = "&#10003;" if step["status"] == "SUCCESS" else "&#10007;"
            agent_display = step["agent"].replace("_", " ").title()

            node = f"""
            <div class="node" style="background:{color}; border-color:{border_color};"
                 onclick="selectNode({i})" id="node-{i}">
                <div class="node-icon">{status_icon}</div>
                <div class="node-name">{agent_display}</div>
                <div class="node-tool">{step['tool']}</div>
                <div class="node-duration">{step['duration']}s</div>
            </div>
            """
            if i < len(steps) - 1:
                node += '<div class="arrow">&#8594;</div>'
            nodes_html_parts.append(node)

        nodes_joined = "\n".join(nodes_html_parts)

        graph_html = f"""
        <style>
            .flow-container {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 30px 10px;
                gap: 0;
                flex-wrap: wrap;
            }}
            .node {{
                border: 3px solid;
                border-radius: 12px;
                padding: 16px 20px;
                min-width: 140px;
                text-align: center;
                color: white;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            .node:hover {{
                transform: scale(1.08);
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            }}
            .node.selected {{
                transform: scale(1.08);
                box-shadow: 0 0 0 4px #0d6efd, 0 4px 16px rgba(0,0,0,0.3);
            }}
            .node-icon {{
                font-size: 24px;
                margin-bottom: 4px;
            }}
            .node-name {{
                font-weight: 700;
                font-size: 14px;
                margin-bottom: 2px;
            }}
            .node-tool {{
                font-size: 11px;
                opacity: 0.85;
            }}
            .node-duration {{
                font-size: 11px;
                opacity: 0.7;
                margin-top: 4px;
            }}
            .arrow {{
                font-size: 28px;
                color: #666;
                padding: 0 8px;
            }}
        </style>

        <div class="flow-container">
            {nodes_joined}
        </div>

        <script>
            function selectNode(idx) {{
                // Remove selected class from all nodes
                document.querySelectorAll('.node').forEach(n => n.classList.remove('selected'));
                // Add to clicked node
                document.getElementById('node-' + idx).classList.add('selected');

                // Send selection to Streamlit
                const streamlitData = {{selected_node: idx}};
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    data: idx
                }}, '*');
            }}
        </script>
        """

        components.html(graph_html, height=180)

        # Agent selector (fallback for interaction since html component messages
        # don't integrate natively — use selectbox)
        st.markdown("---")
        agent_names = [s["agent"] for s in steps]
        selected_agent = st.selectbox(
            "Select an agent to view details:",
            agent_names,
            format_func=lambda x: x.replace("_", " ").title()
        )

        if selected_agent:
            selected_step = next((s for s in steps if s["agent"] == selected_agent), None)
            if selected_step:
                status_icon = "✅" if selected_step["status"] == "SUCCESS" else "❌"

                st.markdown(f"### {status_icon} {selected_agent.replace('_', ' ').title()}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Tool", selected_step["tool"])
                c2.metric("Duration", f"{selected_step['duration']}s")
                c3.metric("Status", selected_step["status"])

                if selected_step.get("reason"):
                    st.markdown(f"*{selected_step['reason']}*")

                col_in, col_out = st.columns(2)
                with col_in:
                    st.markdown("**Input Arguments**")
                    st.json(selected_step["arguments"])
                with col_out:
                    st.markdown("**Output Result**")
                    st.json(selected_step["result"])

                # Show connection to next agent
                step_idx = steps.index(selected_step)
                if step_idx < len(steps) - 1:
                    next_step = steps[step_idx + 1]
                    st.markdown("---")
                    st.markdown(
                        f"**Next →** {next_step['agent'].replace('_', ' ').title()} "
                        f"(`{next_step['tool']}`)"
                    )
                    st.markdown("**Data passed to next agent:**")
                    st.json(next_step["arguments"])
