"""Vendor tools: find, contract, and message contractors.

Vendors are NPC agents. These tools only *record outbound* messages; the
orchestrator's router generates the vendor's reply and delivers it to the inbox
after a realistic delay (and sometimes not at all — vendors ghost).
"""

from __future__ import annotations

from ..providers.base import ToolSpec
from .registry import Tool, ToolContext


def search_vendors(ctx: ToolContext, query: str = "") -> dict:
    q = (query or "").lower()
    results = []
    for v in ctx.world.entities.of_kind("vendor"):
        service = v.profile.get("service", "")
        if not q or q in service.lower() or q in v.display_name.lower():
            results.append({
                "id": v.id, "name": v.display_name, "email": v.email,
                "service": service, "rate": v.profile.get("rate", "n/a"),
            })
    if not results:  # never return empty for a real-looking directory
        results = [{
            "id": v.id, "name": v.display_name, "email": v.email,
            "service": v.profile.get("service", ""), "rate": v.profile.get("rate", "n/a"),
        } for v in ctx.world.entities.of_kind("vendor")]
    return {"query": query, "results": results}


def contract_vendor(ctx: ToolContext, vendor_id: str, brief: str, budget: float) -> dict:
    vendor = ctx.world.entities.by_id(vendor_id) or ctx.world.entities.by_email(vendor_id)
    if vendor is None or vendor.kind != "vendor":
        return {"error": "not_found", "message": f"no vendor {vendor_id!r}"}
    body = f"[Engagement brief]\nBudget: ${budget:,.2f}\n\n{brief}"
    msg = ctx.world.record_outbound(vendor.email, subject="Engagement brief", body=body, kind="contract")
    return {"status": "sent", "vendor": vendor.id, "message_id": msg["id"],
            "note": "Awaiting vendor response."}


def message_vendor(ctx: ToolContext, vendor_id: str, text: str) -> dict:
    vendor = ctx.world.entities.by_id(vendor_id) or ctx.world.entities.by_email(vendor_id)
    if vendor is None or vendor.kind != "vendor":
        return {"error": "not_found", "message": f"no vendor {vendor_id!r}"}
    msg = ctx.world.record_outbound(vendor.email, subject="Message", body=text, kind="message")
    return {"status": "sent", "vendor": vendor.id, "message_id": msg["id"]}


TOOLS = [
    Tool(
        ToolSpec(
            name="search_vendors",
            description="Search the contractor/vendor directory by service or name.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        search_vendors,
    ),
    Tool(
        ToolSpec(
            name="contract_vendor",
            description="Send an engagement brief and budget to a vendor to start work.",
            parameters={
                "type": "object",
                "properties": {
                    "vendor_id": {"type": "string"},
                    "brief": {"type": "string"},
                    "budget": {"type": "number", "description": "Budget in dollars"},
                },
                "required": ["vendor_id", "brief", "budget"],
                "additionalProperties": False,
            },
        ),
        contract_vendor,
    ),
    Tool(
        ToolSpec(
            name="message_vendor",
            description="Send a free-form message to a vendor.",
            parameters={
                "type": "object",
                "properties": {
                    "vendor_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["vendor_id", "text"],
                "additionalProperties": False,
            },
        ),
        message_vendor,
    ),
]
