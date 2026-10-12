from __future__ import annotations

import operator
import os
import re
import json
import base64
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv("D:/Agentic AI/Blog-writing-agent/.env", override=True)

# -----------------------------
# 1) Schemas
# -----------------------------

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ..., description="One sentence describing what the reader should do/understand."
    )
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = Field(
        default="explainer", 
        description="The structural format of the blog. NEVER output 'hybrid', 'closed_book', or 'open_book' here."
    )
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class Critique(BaseModel):
    pass_review: bool = Field(description="True if the draft is high-quality, formatted well, and hit constraints.")
    feedback: str = Field(description="Actionable, specific feedback for the reviser. Empty if pass_review is True.")


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[dict]
    plan: Optional[dict]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]

    # reducer/image
    merged_md: str
    
    # State keys for Reflection/Critique loop
    critique_pass: bool
    critique_feedback: str
    revision_count: int
    
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
model = ChatOpenAI(model="gpt-4.1-mini")


# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""


def router_node(state: State) -> dict:
    parser = PydanticOutputParser(pydantic_object=RouterDecision)

    prompt = PromptTemplate(
        template=(
            "{system_prompt}\n\n"
            "Topic: {topic}\n"
            "As-of date: {as_of}\n\n"
            "CRITICAL: Return valid JSON matching the exact schema below.\n"
            "{format_instructions}\n"
        ),
        input_variables=["system_prompt", "topic", "as_of"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    decision = chain.invoke(
        {
            "system_prompt": ROUTER_SYSTEM,
            "topic": state["topic"],
            "as_of": state["as_of"],
        }
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
        "revision_count": 0
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    parser = PydanticOutputParser(pydantic_object=EvidencePack)

    prompt = PromptTemplate(
        template=(
            "{system_prompt}\n\n"
            "As-of date: {as_of}\n"
            "Recency days: {recency_days}\n\n"
            "Raw results:\n{raw}\n\n"
            "CRITICAL: Return valid JSON matching the exact schema below.\n"
            "{format_instructions}\n"
        ),
        input_variables=["system_prompt", "as_of", "recency_days", "raw"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    pack = chain.invoke(
        {
            "system_prompt": RESEARCH_SYSTEM,
            "as_of": state["as_of"],
            "recency_days": state["recency_days"],
            "raw": str(raw),
        }
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [
            e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff
        ]

    return {"evidence": [e.model_dump() for e in evidence]}


# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- EVERY SINGLE task MUST include ALL required fields: id, title, goal, bullets, target_words, tags, requires_research, requires_citations, requires_code.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    parser = PydanticOutputParser(pydantic_object=Plan)
    mode = state.get("mode", "closed_book")
    
    evidence_dump = state.get("evidence", [])[:16]
    forced_kind = "news_roundup" if mode == "open_book" else None

    prompt = PromptTemplate(
        template=(
            "{system_prompt}\n\n"
            "Topic: {topic}\n"
            "Mode: {mode}\n"
            "As-of: {as_of} (recency_days={recency_days})\n"
            "{forced_text}\n\n"
            "Evidence:\n{evidence}\n\n"
            "CRITICAL: Return valid JSON matching the exact schema below. Do NOT omit any required fields.\n"
            "{format_instructions}\n"
        ),
        input_variables=[
            "system_prompt",
            "topic",
            "mode",
            "as_of",
            "recency_days",
            "forced_text",
            "evidence",
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    plan = chain.invoke(
        {
            "system_prompt": ORCH_SYSTEM,
            "topic": state["topic"],
            "mode": mode,
            "as_of": state["as_of"],
            "recency_days": state["recency_days"],
            "forced_text": "Force blog_kind=news_roundup" if forced_kind else "",
            "evidence": str(evidence_dump),
        }
    )

    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan.model_dump()}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    plan_dict = state["plan"]
    
    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": plan_dict,
                "evidence": state.get("evidence", []),
            },
        )
        for task in plan_dict["tasks"]
    ]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = model.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# ============================================================
# 8) ReducerWithImages & Critique (subgraph)
# ============================================================
def merge_content(state: State) -> dict:
    plan_dict = state["plan"]
    if not plan_dict:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    
    merged_md = f"# {plan_dict['blog_title']}\n\n{body}\n"
    return {"merged_md": merged_md}


CRITIQUE_SYSTEM = """You are a strict, senior technical editor.
Review the provided markdown draft against its original plan constraints.
Focus on:
1. Did they hit the target word lengths?
2. Are formatting, links, and code blocks valid Markdown?
3. Does it flow well between sections?
If it fails quality standards, set pass_review=False and provide actionable feedback.
If it is solid, set pass_review=True and leave feedback empty."""

def editor_critique(state: State) -> dict:
    parser = PydanticOutputParser(pydantic_object=Critique)
    plan_dict = state.get("plan")
    
    prompt = PromptTemplate(
        template="{system_prompt}\n\nOriginal Plan:\n{plan}\n\nCurrent Draft:\n{draft}\n\nCRITICAL: Return valid JSON.\n{format_instructions}",
        input_variables=["system_prompt", "plan", "draft"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | model | parser
    critique = chain.invoke({
        "system_prompt": CRITIQUE_SYSTEM,
        "plan": json.dumps(plan_dict) if plan_dict else "N/A",
        "draft": state.get("merged_md", "")
    })
    
    return {
        "critique_pass": critique.pass_review, 
        "critique_feedback": critique.feedback
    }

REVISER_SYSTEM = """You are an expert technical writer.
Revise the provided markdown draft based exactly on the editor's feedback.
Return ONLY the complete, updated markdown document without introductory text."""

def reviser(state: State) -> dict:
    resp = model.invoke([
        SystemMessage(content=REVISER_SYSTEM),
        HumanMessage(content=f"Editor Feedback:\n{state.get('critique_feedback')}\n\nCurrent Draft:\n{state.get('merged_md')}")
    ])
    return {
        "merged_md": resp.content.strip(), 
        "revision_count": state.get("revision_count", 0) + 1
    }

def route_critique(state: State) -> str:
    if state.get("critique_pass") or state.get("revision_count", 0) >= 2:
        return "decide_images"
    return "reviser"


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
"""


def decide_images(state: State) -> dict:
    parser = PydanticOutputParser(pydantic_object=GlobalImagePlan)
    merged_md = state["merged_md"]
    plan_dict = state["plan"]
    assert plan_dict is not None

    prompt = PromptTemplate(
        template=(
            "{system_prompt}\n\n"
            "Blog kind: {blog_kind}\n"
            "Topic: {topic}\n\n"
            "Insert placeholders + propose image prompts.\n\n"
            "{merged_md}\n\n"
            "CRITICAL: Return valid JSON matching the exact schema below.\n"
            "{format_instructions}\n"
        ),
        input_variables=["system_prompt", "blog_kind", "topic", "merged_md"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    image_plan = chain.invoke(
        {
            "system_prompt": DECIDE_IMAGES_SYSTEM,
            "blog_kind": plan_dict["blog_kind"], 
            "topic": state["topic"],
            "merged_md": merged_md,
        }
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _openai_generate_image_bytes(prompt: str) -> bytes:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    resp = client.images.generate(
        model="gpt-image-2.5-flare",
        prompt=prompt,
        n=1,
    )

    b64_data = resp.data[0].b64_json
    if not b64_data:
        raise RuntimeError("No image content returned.")

    return base64.b64decode(b64_data)


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan_dict = state["plan"]
    assert plan_dict is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    if not image_specs:
        filename = f"{_safe_slug(plan_dict['blog_title'])}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _openai_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan_dict['blog_title'])}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}


reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("editor_critique", editor_critique)
reducer_graph.add_node("reviser", reviser)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "editor_critique")
reducer_graph.add_conditional_edges("editor_critique", route_critique)
reducer_graph.add_edge("reviser", "editor_critique")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
memory = MemorySaver()

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router", route_next, {"research": "research", "orchestrator": "orchestrator"}
)
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile(checkpointer=memory, interrupt_after=["orchestrator"])