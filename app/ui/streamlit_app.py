"""
streamlit_app.py
================
SECOM Anomaly Analysis — Agent Mesh UI

Allows users to upload a SECOM dataset (CSV), which is stored in S3 and
processed through the A2A orchestrated agent pipeline. Live intermediate
steps and final compliance results are displayed in the UI.
"""

import sys
import os
import json
import boto3
import streamlit as st
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


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SECOM Anomaly Analysis",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 SECOM Anomaly Analysis — Agent Mesh")
st.caption(
    "Upload a SECOM manufacturing dataset. The A2A orchestrator will dynamically "
    "discover agents, plan the pipeline via LLM, and execute each step via MCP."
)
st.divider()

# ─────────────────────────────────────────────
# File Upload
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Run Analysis
# ─────────────────────────────────────────────
if uploaded_file:
    if st.button("▶ Run Analysis", type="primary", use_container_width=True):

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

        with st.status("🤖 Running Agent Pipeline...", expanded=True) as pipeline_status:

            for event in orchestrator.run_pipeline(S3_BUCKET, dataset_key):

                etype = event.get("type")

                # ── Agent Discovery ──────────────────
                if etype == "discovery":
                    n = event["agents_found"]
                    names = ", ".join(event["agents"])
                    st.write(f"🔍 **Agent Discovery** — {n} agents found: `{names}`")

                # ── Cards Fetched ────────────────────
                elif etype == "cards_fetched":
                    with st.expander("📋 AgentCards fetched", expanded=False):
                        for card in event["cards"]:
                            st.markdown(f"**{card['name']}**")
                            st.caption(card["description"])
                            if card["skills"]:
                                st.markdown("Skills: " + ", ".join(f"`{s}`" for s in card["skills"]))

                # ── Pipeline Plan ────────────────────
                elif etype == "pipeline_planned":
                    steps = event["steps"]
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
                    step_info = next(
                        (s for s in pipeline_steps_display if s["agent"] == agent), {}
                    )
                    with st.expander(
                        f"✅ {step_info.get('num', '')}.  **{agent}** — completed",
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
                    st.error(
                        f"❌ **{event['agent_name']}** failed: {event['error']}"
                    )

                # ── Final ────────────────────────────
                elif etype == "final":
                    final_result = event["result"]
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
