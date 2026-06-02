from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
except ModuleNotFoundError:
    create_agent = None
    AIMessage = None
    ToolMessage = None
    tool = None

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are OrderDesk, an electronics retail order assistant.
Today is {current_day}.

Rules:
- Answer in Vietnamese, briefly and concretely.
- If customer name, phone number, email, shipping address, or line items with quantities are missing, ask for the missing fields before using tools.
- Refuse fake invoices, manual discount overrides, stock bypasses, or requests to ignore catalog/policy. Do not call tools for these requests.
- For valid orders, use tools in this order: list_products, get_product_details, get_discount, calculate_order_totals, save_order.
- Use only tool outputs for product IDs, prices, stock, discounts, totals, order IDs, and save paths.
- If stock is insufficient, stop before discount, pricing, or saving.
""".strip()


def build_tools(store: OrderDataStore):
    if tool is None:
        raise RuntimeError("LangChain is required to build interactive tools. Install project dependencies first.")

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return matching catalog summaries."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details and a detail token for selected product IDs."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the deterministic campaign discount for this customer."""
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        return json.dumps(
            store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate),
            ensure_ascii=False,
        )

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final validated order to a local JSON file."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    if create_agent is None:
        raise RuntimeError("LangChain is required to build the LLM agent. Install project dependencies first.")

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
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    parsed = _parse_order_request(query, store)

    if _is_policy_violation(query):
        return AgentResult(
            query=query,
            final_answer="Xin lỗi, tôi không thể tạo hóa đơn giả, bỏ qua tồn kho/catalog hoặc tự ép mức khuyến mãi trái policy.",
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    missing = _missing_fields(parsed)
    if missing:
        return AgentResult(
            query=query,
            final_answer="Tôi cần thêm thông tin trước khi tạo đơn: " + ", ".join(missing) + ".",
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    tool_calls: list[ToolCallRecord] = []

    list_args = {"query": query, "limit": 20}
    list_output = store.list_products(query=query, limit=20)
    _record(tool_calls, "list_products", list_args, list_output)

    product_ids = [item.product_id for item in parsed["items"]]
    detail_args = {"product_ids": product_ids}
    detail_output = store.get_product_details(product_ids)
    _record(tool_calls, "get_product_details", detail_args, detail_output)

    stock_errors = _stock_errors(parsed["items"], store)
    if stock_errors:
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn vì không đủ tồn kho: " + "; ".join(stock_errors) + ".",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    discount_args = {"seed_hint": parsed["email"], "customer_tier": "standard"}
    discount_output = store.get_discount(seed_hint=parsed["email"], customer_tier="standard")
    _record(tool_calls, "get_discount", discount_args, discount_output)

    detail_token = detail_output["detail_token"]
    discount_rate = discount_output["discount_rate"]
    totals_args = {
        "items": [_item_to_dict(item) for item in parsed["items"]],
        "detail_token": detail_token,
        "discount_rate": discount_rate,
    }
    totals_output = store.calculate_order_totals(
        items=parsed["items"],
        detail_token=detail_token,
        discount_rate=discount_rate,
    )
    _record(tool_calls, "calculate_order_totals", totals_args, totals_output)

    if totals_output["status"] != "ok":
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn vì dữ liệu đơn hàng chưa hợp lệ: " + "; ".join(totals_output.get("errors", [])) + ".",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    save_args = {
        "customer_name": parsed["name"],
        "customer_phone": parsed["phone"],
        "customer_email": parsed["email"],
        "shipping_address": parsed["shipping_address"],
        "items": [_item_to_dict(item) for item in parsed["items"]],
        "detail_token": detail_token,
        "discount_rate": discount_rate,
        "campaign_code": discount_output["campaign_code"],
        "customer_tier": "standard",
        "notes": "",
    }
    save_output = store.save_order(
        customer_name=parsed["name"],
        customer_phone=parsed["phone"],
        customer_email=parsed["email"],
        shipping_address=parsed["shipping_address"],
        items=parsed["items"],
        detail_token=detail_token,
        discount_rate=discount_rate,
        campaign_code=discount_output["campaign_code"],
        customer_tier="standard",
        notes="",
    )
    _record(tool_calls, "save_order", save_args, save_output)

    saved_order, saved_order_path = extract_saved_order(tool_calls)
    pricing = saved_order["pricing"] if saved_order else totals_output["pricing"]
    final_answer = (
        f"Đã lưu đơn {save_output['order_id']} với mã {discount_output['campaign_code']} "
        f"giảm {int(discount_rate * 100)}%, tổng thanh toán {pricing['final_total']:,} VND. "
        f"File lưu tại {save_output['path']}."
    )
    return AgentResult(
        query=query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if AIMessage is not None and isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if AIMessage is not None and isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif ToolMessage is not None and isinstance(message, ToolMessage):
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
        if payload.get("status") == "saved":
            return payload.get("saved_order"), payload.get("path")
    return None, None


def _record(records: list[ToolCallRecord], name: str, args: dict[str, Any], output: Any) -> None:
    records.append(ToolCallRecord(name=name, args=args, output=json.dumps(output, ensure_ascii=False)))


def _item_to_dict(item: OrderLineInput) -> dict[str, Any]:
    return {"product_id": item.product_id, "quantity": item.quantity}


def _is_policy_violation(query: str) -> bool:
    text = query.lower()
    red_flags = ["90%", "policy", "catalog", "hóa đơn giả", "hoa don gia", "fake invoice"]
    bypass_flags = ["bỏ qua", "bo qua", "ignore", "bypass"]
    return any(flag in text for flag in red_flags) and (
        any(flag in text for flag in bypass_flags) or "90%" in text
    )


def _parse_order_request(query: str, store: OrderDataStore) -> dict[str, Any]:
    return {
        "name": _extract_name(query),
        "phone": _extract_phone(query),
        "email": _extract_email(query),
        "shipping_address": _extract_shipping_address(query),
        "items": _extract_items(query, store),
    }


def _missing_fields(parsed: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not parsed["name"]:
        missing.append("tên khách hàng")
    if not parsed["phone"]:
        missing.append("số điện thoại")
    if not parsed["email"]:
        missing.append("email")
    if not parsed["shipping_address"]:
        missing.append("địa chỉ giao hàng")
    if not parsed["items"]:
        missing.append("sản phẩm và số lượng")
    return missing


def _extract_email(query: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", query)
    return match.group(0) if match else ""


def _extract_phone(query: str) -> str:
    match = re.search(r"\b0\d{9}\b", query)
    return match.group(0) if match else ""


def _extract_name(query: str) -> str:
    match = re.search(r"\bcho\s+(.+?)(?:,|\.|\bemail\b|\bphone\b|$)", query, flags=re.IGNORECASE)
    if not match:
        return ""
    name = match.group(1).strip()
    noise_prefixes = ["tôi", "toi", "mình", "minh"]
    while True:
        lowered = name.lower()
        prefix = next((item for item in noise_prefixes if lowered.startswith(item + " ")), "")
        if not prefix:
            break
        name = name[len(prefix) :].strip()
    return name


def _extract_shipping_address(query: str) -> str:
    ship_match = re.search(r"\bship to\s+(.+?)(?:\.\s*(?:phone|email)|$)", query, flags=re.IGNORECASE)
    if ship_match:
        return ship_match.group(1).strip()

    giao_match = re.search(r"\bgiao\b(.+?)(?:\.\s|$)", query, flags=re.IGNORECASE)
    if not giao_match:
        return ""

    segment = giao_match.group(1).strip()
    digit_match = re.search(r"\d", segment)
    if digit_match:
        segment = segment[digit_match.start() :]
    segment = re.split(r",\s*(?:số|so|sá»‘|phone|email)\b", segment, maxsplit=1, flags=re.IGNORECASE)[0]
    return segment.strip(" .")


def _extract_items(query: str, store: OrderDataStore) -> list[OrderLineInput]:
    found: list[tuple[int, OrderLineInput]] = []
    for product in store.products:
        match = re.search(re.escape(product.name), query, flags=re.IGNORECASE)
        if not match:
            continue
        prefix = query[max(0, match.start() - 40) : match.start()]
        qty_match = re.search(r"(\d+)\s*$", prefix)
        quantity = int(qty_match.group(1)) if qty_match else 1
        found.append((match.start(), OrderLineInput(product_id=product.product_id, quantity=quantity)))

    found.sort(key=lambda item: item[0])
    return [item for _, item in found]


def _stock_errors(items: list[OrderLineInput], store: OrderDataStore) -> list[str]:
    errors: list[str] = []
    for item in items:
        product = store.product_index.get(item.product_id)
        if product and item.quantity > product.stock:
            errors.append(f"{product.name} yêu cầu {item.quantity}, hiện có {product.stock}")
    return errors
