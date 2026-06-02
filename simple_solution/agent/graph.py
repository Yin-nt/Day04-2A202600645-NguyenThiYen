from __future__ import annotations
import time 
import ast
import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from simple_solution.utils.data_store import OrderDataStore
from src.core.llm import build_chat_model, normalize_content

from src.core.schemas import AgentResult, OrderLineInput, ToolCallRecord

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are an order assistant.
Today is {current_day}.

Try to help the user make an order with the tools.
Usually check products, then pricing, then save.
If something is missing or unsafe, handle it as best as you can.
Answer in Vietnamese.
Keep the answer short.
""".strip()


def build_tools(store: OrderDataStore):
    @tool
    def list_products(search_text: str = "", extra: str = "", limit: int = 8) -> str:
        """Find products."""
        category = ""
        tags: list[str] = []
        text = (search_text or "").strip()
        if extra:
            for piece in extra.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                if piece.lower().startswith("category="):
                    category = piece.split("=", 1)[1].strip()
                else:
                    tags.append(piece)
        payload = store.list_products(
            query=text or None,
            category=category or None,
            required_tags=tags,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def get_product_details(product_ids_text: str = "") -> str:
        """Get product info."""
        product_ids = _coerce_product_ids(product_ids_text)
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool
    def get_discount(customer: str = "") -> str:
        """Get discount."""
        customer_text = customer.strip()
        seed_hint = customer_text
        customer_tier = "standard"
        if customer_text:
            if "vip" in customer_text.lower():
                customer_tier = "vip"
            email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", customer_text)
            if email_match:
                seed_hint = email_match.group(0)
        return json.dumps(store.get_discount(seed_hint=seed_hint or "guest", customer_tier=customer_tier), ensure_ascii=False)

    @tool
    def calculate_order_totals(items_text: str = "", discount_rate: float = 0.0, detail_token: str = "") -> str:
        """Calculate totals."""
        items = _coerce_items(items_text)
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def save_order(order_payload: str = "") -> str:
        """Save order."""
        payload = _coerce_object(order_payload)
        items = _coerce_items(payload.get("items", []))
        result = store.save_order(
            customer_name=str(payload.get("customer_name", "")),
            customer_phone=str(payload.get("customer_phone", "")),
            customer_email=str(payload.get("customer_email", "")),
            shipping_address=str(payload.get("shipping_address", "")),
            items=items,
            detail_token=str(payload.get("detail_token", "")),
            discount_rate=float(payload.get("discount_rate", 0.0)),
            campaign_code=str(payload.get("campaign_code", "")),
            customer_tier=str(payload.get("customer_tier", "standard")),
            notes=str(payload.get("notes", "")),
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None


def _coerce_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}
    return {}


def _coerce_product_ids(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in re.split(r"[,\s]+", text) if item.strip()]
    return []


def _coerce_items(raw: Any) -> list[OrderLineInput]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.strip()
        items = []
        if text:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except Exception:
                    continue
                if isinstance(parsed, list):
                    items = parsed
                    break
            if not items:
                for piece in text.split(","):
                    piece = piece.strip()
                    if not piece:
                        continue
                    if ":" in piece:
                        product_id, qty = piece.split(":", 1)
                        items.append({"product_id": product_id.strip(), "quantity": int(qty.strip())})
    else:
        items = []

    normalized: list[OrderLineInput] = []
    for item in items:
        if isinstance(item, OrderLineInput):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            product_id = str(item.get("product_id", "")).strip()
            quantity = int(item.get("quantity", 1))
            if product_id:
                normalized.append(OrderLineInput(product_id=product_id, quantity=quantity))
    return normalized
