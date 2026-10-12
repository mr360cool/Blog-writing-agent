from __future__ import annotations

import json
import os
import re
import uuid 
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from openai_backend import app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Optional[Dict[str, Any]], config: dict) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        for step in graph_app.stream(inputs, config, stream_mode="updates"):
            yield ("updates", step)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, config, stream_mode="values"):
            yield ("values", step)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs, config)
    yield ("final", out)


def extract_latest_state(
    current_state: Dict[str, Any], step_payload: Any
) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(
            next(iter(step_payload.values())), dict
        ):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            # ✅ FIX: replaced use_container_width=True with width="stretch"
            st.image(src, caption=caption or (alt or None), width="stretch")
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                # ✅ FIX: replaced use_container_width=True with width="stretch"
                st.image(
                    str(img_path),
                    caption=caption or (alt or None),
                    width="stretch",
                )
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# Past blogs helpers (UNCHANGED)
# -----------------------------
def list_past_blogs() -> List[Path]:
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = "idle" 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "current_state" not in st.session_state:
    st.session_state["current_state"] = {}

logs = st.session_state["logs"]

def log(msg: str):
    logs.append(msg)

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area(
        "Topic",
        height=120,
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None, 
                    "evidence": [],  
                    "image_specs": [],  
                    "final": md_text,  
                }
                st.session_state["topic_prefill"] = extract_title_from_md(
                    md_text, selected_md_file.stem
                )
                st.session_state.workflow_status = "idle" 


if "topic_prefill" in st.session_state and isinstance(
    st.session_state["topic_prefill"], str
):
    pass

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

# -----------------------------
# Graph Execution Logic with Pauses
# -----------------------------
config = {"configurable": {"thread_id": st.session_state.thread_id}}

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.workflow_status = "running"
    
    st.session_state.inputs = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }
    st.session_state.current_state = {}
    st.session_state["logs"] = []

# Phase 1 or 3 (Running or Resuming)
if st.session_state.workflow_status in ["running", "resuming"]:
    inputs_to_pass = st.session_state.inputs if st.session_state.workflow_status == "running" else None
    
    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()
    last_node = None

    for kind, payload in try_stream(app, inputs_to_pass, config):
        if kind in ("updates", "values"):
            node_name = None
            if (
                isinstance(payload, dict)
                and len(payload) == 1
                and isinstance(next(iter(payload.values())), dict)
            ):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            st.session_state.current_state = extract_latest_state(st.session_state.current_state, payload)

            summary = {
                "mode": st.session_state.current_state.get("mode"),
                "needs_research": st.session_state.current_state.get("needs_research"),
                "queries": (
                    st.session_state.current_state.get("queries", [])[:5]
                    if isinstance(st.session_state.current_state.get("queries"), list)
                    else []
                ),
                "evidence_count": len(st.session_state.current_state.get("evidence", []) or []),
                "tasks": (
                    len((st.session_state.current_state.get("plan") or {}).get("tasks", []))
                    if isinstance(st.session_state.current_state.get("plan"), dict)
                    else None
                ),
                "images": len(st.session_state.current_state.get("image_specs", []) or []),
                "sections_done": len(st.session_state.current_state.get("sections", []) or []),
                "revisions_done": st.session_state.current_state.get("revision_count", 0)
            }
            progress_area.json(summary)
            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

    current_graph_state = app.get_state(config)
    if current_graph_state.next:
        st.session_state.workflow_status = "pending_approval"
        st.rerun()
    else:
        st.session_state.workflow_status = "completed"
        st.session_state["last_out"] = current_graph_state.values
        status.update(label="✅ Done", state="complete", expanded=False)
        st.rerun()

# Phase 2 (Human-in-the-Loop Plan Approval)
if st.session_state.workflow_status == "pending_approval":
    st.warning("⏸️ Agent Paused: Human-in-the-Loop Plan Approval")
    current_graph_state = app.get_state(config)
    
    # ✅ FIX: plan_dict is natively a dictionary now
    plan_dict = current_graph_state.values.get("plan")
    
    if plan_dict:
        st.subheader("Edit Outline / Task Definitions")
        st.markdown("Modify the titles, goals, and target word counts below. When satisfied, click **Approve** to fan-out to the worker nodes.")
        
        tasks = plan_dict.get("tasks", [])
        
        df = pd.DataFrame(tasks)
        # ✅ FIX: replaced use_container_width=True with width="stretch"
        edited_df = st.data_editor(df, width="stretch", disabled=["id"])
        
        if st.button("✅ Approve Plan & Continue Generating", type="primary"):
            updated_tasks = edited_df.to_dict('records')
            
            plan_dict["tasks"] = updated_tasks
                
            app.update_state(config, {"plan": plan_dict})
            st.session_state.workflow_status = "resuming"
            st.rerun()

# -----------------------------
# Results Dashboard rendering
# -----------------------------
if st.session_state.workflow_status in ["idle", "completed"]:
    out = st.session_state.get("last_out")
    if out:
        # --- Plan tab ---
        with tab_plan:
            st.subheader("Plan")
            plan_dict = out.get("plan") # Plan is now a native dict
            
            if not plan_dict:
                st.info("No plan found in output.")
            else:
                st.write("**Title:**", plan_dict.get("blog_title"))
                cols = st.columns(3)
                cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
                cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
                cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

                tasks = plan_dict.get("tasks", [])
                if tasks:
                    df = pd.DataFrame(
                        [
                            {
                                "id": t.get("id"),
                                "title": t.get("title"),
                                "target_words": t.get("target_words"),
                                "requires_research": t.get("requires_research"),
                                "requires_citations": t.get("requires_citations"),
                                "requires_code": t.get("requires_code"),
                                "tags": ", ".join(t.get("tags") or []),
                            }
                            for t in tasks
                        ]
                    ).sort_values("id")
                    
                    # ✅ FIX: replaced use_container_width=True with width="stretch"
                    st.dataframe(df, width="stretch", hide_index=True)

                    with st.expander("Task details"):
                        st.json(tasks)

        # --- Evidence tab ---
        with tab_evidence:
            st.subheader("Evidence")
            evidence = out.get("evidence") or []
            if not evidence:
                st.info(
                    "No evidence returned (maybe closed_book mode or no Tavily key/results)."
                )
            else:
                rows = []
                for e in evidence:
                    # evidence contains dicts now
                    rows.append(
                        {
                            "title": e.get("title"),
                            "published_at": e.get("published_at"),
                            "source": e.get("source"),
                            "url": e.get("url"),
                        }
                    )
                # ✅ FIX: replaced use_container_width=True with width="stretch"
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # --- Preview tab ---
        with tab_preview:
            st.subheader("Markdown Preview")
            
            revisions = out.get("revision_count", 0)
            if revisions > 0:
                st.success(f"✨ The Agent editor critiqued and revised this draft **{revisions}** times to meet quality standards.")
                with st.expander("Latest Editor Feedback"):
                    st.markdown(f"*{out.get('critique_feedback', 'No feedback recorded.')}*")
            
            final_md = out.get("final") or ""
            if not final_md:
                st.warning("No final markdown found.")
            else:
                render_markdown_with_local_images(final_md)

                plan_dict = out.get("plan") or {}
                blog_title = plan_dict.get("blog_title", extract_title_from_md(final_md, "blog"))

                md_filename = f"{safe_slug(blog_title)}.md"
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                )

                bundle = bundle_zip(final_md, md_filename, Path("images"))
                st.download_button(
                    "📦 Download Bundle (MD + images)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                )

        # --- Images tab ---
        with tab_images:
            st.subheader("Images")
            specs = out.get("image_specs") or []
            images_dir = Path("images")

            if not specs and not images_dir.exists():
                st.info("No images generated for this blog.")
            else:
                if specs:
                    st.write("**Image plan:**")
                    st.json(specs)

                if images_dir.exists():
                    files = [p for p in images_dir.iterdir() if p.is_file()]
                    if not files:
                        st.warning("images/ exists but is empty.")
                    else:
                        for p in sorted(files):
                            # ✅ FIX: replaced use_container_width=True with width="stretch"
                            st.image(str(p), caption=p.name, width="stretch")

                    z = images_zip(images_dir)
                    if z:
                        st.download_button(
                            "⬇️ Download Images (zip)",
                            data=z,
                            file_name="images.zip",
                            mime="application/zip",
                        )

        # --- Logs tab ---
        with tab_logs:
            st.subheader("Logs")
            if logs:
                st.text_area(
                    "Event log", value="\n\n".join(logs[-80:]), height=520
                )