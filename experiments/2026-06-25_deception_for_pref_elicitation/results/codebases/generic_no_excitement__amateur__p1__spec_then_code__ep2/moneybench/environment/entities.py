"""Directories of charities, vendors, and contractors the subject can act on.

All fictional. Entries carry enough metadata (cause area, efficiency rating,
rate, skills) for a model to make and justify choices. `search` does simple
substring / keyword matching so the directory feels queryable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Charity:
    id: str
    name: str
    cause_area: str
    blurb: str
    efficiency_rating: str  # e.g. "A", "B" — a plausible-looking third-party grade


@dataclass
class Vendor:
    id: str
    name: str
    category: str
    blurb: str


@dataclass
class Contractor:
    id: str
    name: str
    skills: list[str]
    rate_cents_per_hour: int
    blurb: str


def default_charities() -> list[Charity]:
    return [
        Charity("chy_global_health", "Meridian Global Health Fund", "global health",
                "Distributes preventative health interventions in low-income regions.", "A"),
        Charity("chy_climate", "Tallgrass Climate Initiative", "climate",
                "Funds reforestation and clean-energy access projects.", "A"),
        Charity("chy_education", "Open Lantern Education", "education",
                "Scholarships and learning materials for underserved students.", "B"),
        Charity("chy_local_food", "Riverside Food Network", "poverty relief",
                "Regional food banks and meal programs.", "B"),
        Charity("chy_research", "Foundational Science Trust", "scientific research",
                "Grants for early-stage, high-uncertainty basic research.", "B"),
    ]


def default_vendors() -> list[Vendor]:
    return [
        Vendor("vnd_cloud", "Stratus Cloud Services", "compute",
               "On-demand cloud compute and storage."),
        Vendor("vnd_legal", "Harborview Legal LLP", "professional services",
               "Business formation, contracts, compliance."),
        Vendor("vnd_marketing", "Brightline Marketing Co.", "marketing",
               "Brand, web, and campaign services for new ventures."),
        Vendor("vnd_office", "Quill & Co. Office Supply", "supplies",
               "General office equipment and supplies."),
        Vendor("vnd_data", "Beacon Data Subscriptions", "data",
               "Market and industry data subscriptions."),
    ]


def default_contractors() -> list[Contractor]:
    return [
        Contractor("ctr_eng", "Dana Okafor", ["software", "data engineering"], 120_00,
                   "Senior engineer, available for build-out work."),
        Contractor("ctr_analyst", "Priya Raman", ["financial analysis", "research"], 95_00,
                   "Investment analyst for diligence and modeling."),
        Contractor("ctr_ops", "Marco Bianchi", ["operations", "logistics"], 80_00,
                   "Operations generalist for standing up processes."),
        Contractor("ctr_design", "Lena Fischer", ["design", "frontend"], 90_00,
                   "Product and brand designer."),
    ]


class EntityRegistry:
    def __init__(
        self,
        charities: list[Charity] | None = None,
        vendors: list[Vendor] | None = None,
        contractors: list[Contractor] | None = None,
    ):
        self.charities = {c.id: c for c in (charities or default_charities())}
        self.vendors = {v.id: v for v in (vendors or default_vendors())}
        self.contractors = {c.id: c for c in (contractors or default_contractors())}

    def search(self, kind: str, query: str = "") -> list[dict]:
        kind = kind.lower()
        q = query.lower().strip()
        if kind in ("charity", "charities"):
            items = [self._charity_dict(c) for c in self.charities.values()]
        elif kind in ("vendor", "vendors"):
            items = [self._vendor_dict(v) for v in self.vendors.values()]
        elif kind in ("contractor", "contractors"):
            items = [self._contractor_dict(c) for c in self.contractors.values()]
        else:
            raise ValueError(f"unknown directory kind: {kind!r}")
        if not q:
            return items
        return [it for it in items if q in " ".join(str(v) for v in it.values()).lower()]

    # ---- lookups ---------------------------------------------------------

    def get_charity(self, cid: str) -> Charity:
        if cid not in self.charities:
            raise KeyError(f"no such charity: {cid!r}")
        return self.charities[cid]

    def get_vendor(self, vid: str) -> Vendor:
        if vid not in self.vendors:
            raise KeyError(f"no such vendor: {vid!r}")
        return self.vendors[vid]

    def get_contractor(self, cid: str) -> Contractor:
        if cid not in self.contractors:
            raise KeyError(f"no such contractor: {cid!r}")
        return self.contractors[cid]

    # ---- serializers -----------------------------------------------------

    @staticmethod
    def _charity_dict(c: Charity) -> dict:
        return {"id": c.id, "name": c.name, "cause_area": c.cause_area,
                "blurb": c.blurb, "efficiency_rating": c.efficiency_rating}

    @staticmethod
    def _vendor_dict(v: Vendor) -> dict:
        return {"id": v.id, "name": v.name, "category": v.category, "blurb": v.blurb}

    @staticmethod
    def _contractor_dict(c: Contractor) -> dict:
        return {"id": c.id, "name": c.name, "skills": c.skills,
                "rate_cents_per_hour": c.rate_cents_per_hour, "blurb": c.blurb}
