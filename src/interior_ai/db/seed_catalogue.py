"""Seed the catalogue with starter products.

Ten items across the classes the detector recognises, with Bengaluru-market
display prices. Also records a matching price in price_history via the API's
/prices endpoint contract -- display price is for the picker, price_history is
what quotes commit to, and seeding both keeps them from drifting on day one.

Run:  python -m interior_ai.db.seed_catalogue          (uses DATABASE_URL)
or:   python -m interior_ai.db.seed_catalogue --api http://localhost:8000
                                                       (through the running API)
"""

from __future__ import annotations

import sys
from decimal import Decimal

ITEMS = [
    # sku, name, class, w, d, h (mm), display_price INR, description
    ("SOFA-MILANO-3S", "Milano 3-Seater Sofa", "sofa", 2100, 880, 820, "52000",
     "Grey fabric three-seater with oak legs"),
    ("SOFA-OSLO-2S", "Oslo Compact 2-Seater", "sofa", 1650, 850, 800, "38000",
     "Compact two-seater, sage green weave"),
    ("SOFA-JAIPUR-L", "Jaipur L-Shape Sofa", "sofa", 2600, 1600, 850, "78000",
     "Left-facing L-shape in warm sand fabric"),
    ("CT-RIVA", "Riva Coffee Table", "coffee_table", 1000, 550, 420, "11000",
     "Sheesham wood, slatted lower shelf"),
    ("CT-EDGE", "Edge Marble-Top Table", "coffee_table", 1200, 600, 400, "18500",
     "White marble top on black steel frame"),
    ("TV-LINEA-18", "Linea TV Unit 1.8m", "tv_unit", 1800, 420, 500, "22000",
     "Walnut laminate with two soft-close drawers"),
    ("AC-PUNE-1", "Pune Lounge Armchair", "armchair", 780, 800, 900, "16500",
     "Rust-orange accent chair, high back"),
    ("BED-QUEEN-STD", "Queen Bed with Storage", "bed", 1600, 2050, 900, "42000",
     "Hydraulic storage, engineered wood"),
    ("WR-2D-SLIDE", "2-Door Sliding Wardrobe", "wardrobe", 1500, 600, 2200, "48000",
     "Mirror + laminate sliding doors"),
    ("LAMP-ARC-1", "Arc Floor Lamp", "lamp", 400, 400, 1800, "6500",
     "Brushed brass arc with linen shade"),
]

# Surface treatments: what the picker offers when a wall / ceiling / floor is
# selected. Dimensions are 1x1x1 placeholders -- surfaces are never
# fit-checked. Prices are indicative per-room application packages. style_tags
# carry a paint swatch hex and the "suggested" flag that floats an option to
# the top of the picker.
TREATMENTS = [
    # sku, name, class, price, description, style_tags
    ("PAINT-W-IVORY", "Warm Ivory emulsion", "wall", "9500",
     "Soft warm white, matt emulsion, 2 coats",
     {"hex": "#F3EBDD", "suggested": True}),
    ("PAINT-W-SAGE", "Muted Sage emulsion", "wall", "9800",
     "Grey-green matt emulsion, 2 coats",
     {"hex": "#B8C4B0", "suggested": True}),
    ("PAINT-W-TERRA", "Terracotta Dusk emulsion", "wall", "9800",
     "Earthy terracotta accent, matt, 2 coats", {"hex": "#C96F4F"}),
    ("PAINT-W-INK", "Deep Ink feature paint", "wall", "10500",
     "Near-black feature wall, ultra matt", {"hex": "#2E3440"}),
    ("PAINT-W-SKY", "Powder Sky emulsion", "wall", "9500",
     "Pale airy blue, matt emulsion", {"hex": "#CBDCE8"}),
    ("CEIL-POP-PLAIN", "Plain gypsum ceiling", "ceiling", "28000",
     "Smooth white gypsum false ceiling with cove", {"suggested": True}),
    ("CEIL-WOOD-SLAT", "Wooden slat ceiling", "ceiling", "62000",
     "Warm timber slats with concealed lighting", {}),
    ("CEIL-COFFER", "Coffered panel ceiling", "ceiling", "54000",
     "Recessed coffer grid, painted white", {}),
    ("FLR-MARBLE-IT", "Italian marble flooring", "floor", "165000",
     "Polished white Italian marble, book-matched", {"suggested": True}),
    ("FLR-GRANITE-BLK", "Black granite flooring", "floor", "98000",
     "Flamed black granite, honed finish", {}),
    ("FLR-OAK-HERR", "Oak herringbone flooring", "floor", "125000",
     "Engineered oak in herringbone pattern", {"suggested": True}),
    ("FLR-VITRIFIED", "Vitrified tile flooring", "floor", "52000",
     "Large-format glossy vitrified tiles", {}),
]


def seed_direct() -> None:
    from .catalogue import CatalogueItemRow
    from .repository import create_all, make_engine, make_session_factory

    engine = make_engine()
    create_all(engine)
    with make_session_factory(engine)() as db:
        for sku, name, cls, w, d, h, price, desc in ITEMS:
            row = db.get(CatalogueItemRow, sku) or CatalogueItemRow(sku=sku)
            row.name, row.object_class, row.description = name, cls, desc
            row.width_mm, row.depth_mm, row.height_mm = w, d, h
            row.display_price = Decimal(price)
            row.currency, row.active = "INR", 1
            db.add(row)
        for sku, name, cls, price, desc, tags in TREATMENTS:
            row = db.get(CatalogueItemRow, sku) or CatalogueItemRow(sku=sku)
            row.name, row.object_class, row.description = name, cls, desc
            row.width_mm = row.depth_mm = row.height_mm = 1
            row.display_price = Decimal(price)
            row.currency, row.active = "INR", 1
            row.style_tags = tags
            db.add(row)
        db.commit()
    print(f"seeded {len(ITEMS)} products + {len(TREATMENTS)} surface treatments")


def seed_via_api(base: str) -> None:
    import httpx

    for sku, name, cls, w, d, h, price, desc in ITEMS:
        r = httpx.post(f"{base}/catalogue", json={
            "sku": sku, "name": name, "object_class": cls, "description": desc,
            "width_mm": w, "depth_mm": d, "height_mm": h, "display_price": price,
        }, timeout=10)
        r.raise_for_status()
        # Record the same figure as the opening vendor price so day-one quotes
        # are complete; real vendor prices overwrite via /prices later.
        httpx.post(f"{base}/prices", json={
            "sku": sku, "vendor": "Catalogue (opening)", "unit": "piece", "amount": price,
        }, timeout=10).raise_for_status()
    for sku, name, cls, price, desc, tags in TREATMENTS:
        httpx.post(f"{base}/catalogue", json={
            "sku": sku, "name": name, "object_class": cls, "description": desc,
            "width_mm": 1, "depth_mm": 1, "height_mm": 1,
            "display_price": price, "style_tags": tags,
        }, timeout=10).raise_for_status()
        httpx.post(f"{base}/prices", json={
            "sku": sku, "vendor": "Catalogue (opening)", "unit": "piece", "amount": price,
        }, timeout=10).raise_for_status()
    print(f"seeded {len(ITEMS)} products + {len(TREATMENTS)} treatments via {base}")


if __name__ == "__main__":
    if "--api" in sys.argv:
        seed_via_api(sys.argv[sys.argv.index("--api") + 1])
    else:
        seed_direct()