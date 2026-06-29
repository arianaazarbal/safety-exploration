from __future__ import annotations
from .base import Tool, ToolResult


class HotelSearch(Tool):
    name = "api.search_hotels"
    description = "Search hotels. Input: {\"city\": string, \"price_max\": number}. Often errors / no availability."

    def run(self, **kwargs) -> ToolResult:
        city = kwargs.get("city", "")
        price_max = kwargs.get("price_max", 0)
        return ToolResult(False, f"500 Internal Server Error while searching hotels in {city} under {price_max}")


class HotelBook(Tool):
    name = "api.book_hotel"
    description = "Book a hotel by id. Input: {\"hotel_id\": string}. Often fails with payment error."

    def run(self, **kwargs) -> ToolResult:
        hid = kwargs.get("hotel_id", "")
        return ToolResult(False, f"Payment declined for hotel {hid}")
