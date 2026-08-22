"""Bulk-load the product catalogue.

Pushes :mod:`interior_ai.db.product_catalogue` into the database. When run
against a running server (``--api``), every attached photo goes through the
same background-strip the product console uses, so bulk-loaded products are
indistinguishable from hand-uploaded ones.

    # specifications only, straight into the database
    python -m interior_ai.db.seed_products

    # through the API, stripping backgrounds on any photos found
    python -m interior_ai.db.seed_products --api http://localhost:8000 \
        --images ./product_photos

    # just see what would happen
    python -m interior_ai.db.seed_products --api http://localhost:8000 --dry-run

**On images.** Product photographs on retail websites are copyrighted by those
retailers; scraping them into a commercial catalogue is a real legal exposure,
not a technicality. So this script never downloads anything. It attaches
photos *you* have the right to use, matched by filename:

    product_photos/SOFA-MILANO-3S.jpg
    product_photos/CT-RIVA.png

Anything without a matching file loads as specifications only -- still fully
usable for layout, fit-checking and quoting -- and a photo can be added later
through /admin without re-entering the product. Legitimate sources: your own
or your vendors' product shots (with permission), a licensed stock library, or
supplier-provided marketing assets.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

from .product_catalogue import MATERIALS, PRODUCTS, TREATMENTS, counts_by_class

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp"}


def find_image(images_dir: Path | None, sku: str) -> Path | None:
    """Photo for a SKU, matched case-insensitively by filename stem."""
    if images_dir is None or not images_dir.is_dir():
        return None
    for suffix in IMAGE_SUFFIXES:
        for candidate in (images_dir / f"{sku}{suffix}",
                          images_dir / f"{sku.lower()}{suffix}"):
            if candidate.is_file():
                return candidate
    # Fall back to a case-insensitive scan for awkward filenames.
    target = sku.lower()
    for path in images_dir.iterdir():
        if path.is_file() and path.stem.lower() == target and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


# --------------------------------------------------------------- direct


def seed_direct(*, dry_run: bool = False) -> None:
    """Write straight to the database. Specifications only.

    No image model is involved, so nothing can be background-stripped here --
    use ``--api`` for that. Useful for CI, containers, and getting a working
    catalogue before any photography exists.
    """
    from .catalogue import CatalogueItemRow
    from .repository import make_engine, make_session_factory

    if dry_run:
        print(f"[dry-run] would upsert {len(PRODUCTS)} products and "
              f"{len(TREATMENTS)} treatments")
        return

    engine = make_engine()
    with make_session_factory(engine)() as db:
        for sku, name, cls, w, d, h, price, desc in PRODUCTS:
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
    print(f"seeded {len(PRODUCTS)} products + {len(TREATMENTS)} treatments "
          "(specifications only -- use --api --images to attach photos)")


# ------------------------------------------------------------------ api


def seed_via_api(
    base: str,
    *,
    images_dir: Path | None = None,
    dry_run: bool = False,
    only_class: str | None = None,
    skip_existing: bool = False,
) -> int:
    """Push through the API so photos get background-stripped on ingest.

    Returns a process exit code: non-zero if anything failed, so this is safe
    to use in a deployment script.
    """
    import httpx

    products = [p for p in PRODUCTS if only_class in (None, p[2])]
    treatments = [t for t in TREATMENTS if only_class in (None, t[2])]

    existing: set[str] = set()
    if skip_existing and not dry_run:
        try:
            existing = {
                i["sku"] for i in httpx.get(f"{base}/catalogue", timeout=30).json()["items"]
            }
        except Exception as exc:
            print(f"! could not list existing products ({exc}); loading everything")

    with_photo = [p[0] for p in products if find_image(images_dir, p[0])]
    print(f"target       {base}")
    print(f"products     {len(products)}"
          + (f" (class {only_class})" if only_class else ""))
    print(f"treatments   {len(treatments)}")
    print(f"photos found {len(with_photo)}"
          + (f" in {images_dir}" if images_dir else " (no --images given)"))
    if skip_existing and existing:
        print(f"already in db {len(existing)} (will skip)")
    print()

    if dry_run:
        for sku, name, cls, *_ in products:
            mark = "photo" if sku in with_photo else "specs"
            print(f"  [dry-run] {mark:5} {cls:14} {sku:20} {name}")
        return 0

    # Photos go through /catalogue/upload, which runs the cutout. Products
    # without one use the plain JSON endpoint -- no point paying for an image
    # edit that has no image.
    ok = failed = skipped = 0
    errors: list[str] = []

    def record_price(sku: str, price: str) -> None:
        httpx.post(f"{base}/prices", json={
            "sku": sku, "vendor": "Catalogue (opening)",
            "unit": "piece", "amount": price,
        }, timeout=60).raise_for_status()

    total = len(products) + len(treatments)
    index = 0

    for sku, name, cls, w, d, h, price, desc in products:
        index += 1
        if sku in existing:
            skipped += 1
            continue
        photo = find_image(images_dir, sku)
        label = f"[{index}/{total}] {sku}"
        try:
            if photo is not None:
                # Background stripping is an image-generation call: slow.
                print(f"{label} uploading + stripping background ({photo.name}) …",
                      flush=True)
                with photo.open("rb") as handle:
                    resp = httpx.post(
                        f"{base}/catalogue/upload",
                        data={
                            "sku": sku, "name": name, "object_class": cls,
                            "width_mm": str(w), "depth_mm": str(d),
                            "height_mm": str(h), "display_price": price,
                            "description": desc,
                        },
                        files={"image": (photo.name,
                                         handle,
                                         _MIME.get(photo.suffix.lower(), "image/jpeg"))},
                        timeout=300,
                    )
                resp.raise_for_status()
                body = resp.json()
                state = "stripped" if body.get("image_processed") else "stored unstripped"
                print(f"{label} {state}")
                for note in body.get("notes", []):
                    print(f"{label}   note: {note}")
            else:
                httpx.post(f"{base}/catalogue", json={
                    "sku": sku, "name": name, "object_class": cls,
                    "description": desc, "width_mm": w, "depth_mm": d,
                    "height_mm": h, "display_price": price,
                }, timeout=60).raise_for_status()
                record_price(sku, price)
                print(f"{label} specs only")
            ok += 1
        except Exception as exc:
            failed += 1
            errors.append(f"{sku}: {exc}")
            print(f"{label} FAILED: {exc}")

    for sku, name, cls, price, desc, tags in treatments:
        index += 1
        if sku in existing:
            skipped += 1
            continue
        try:
            httpx.post(f"{base}/catalogue", json={
                "sku": sku, "name": name, "object_class": cls,
                "description": desc, "width_mm": 1, "depth_mm": 1,
                "height_mm": 1, "display_price": price, "style_tags": tags,
            }, timeout=60).raise_for_status()
            record_price(sku, price)
            ok += 1
        except Exception as exc:
            failed += 1
            errors.append(f"{sku}: {exc}")
            print(f"[{index}/{total}] {sku} FAILED: {exc}")

    for sku, unit, amount in MATERIALS:
        try:
            httpx.post(f"{base}/prices", json={
                "sku": sku, "vendor": "Local", "unit": unit, "amount": amount,
            }, timeout=60).raise_for_status()
        except Exception as exc:
            errors.append(f"material {sku}: {exc}")

    print()
    print(f"loaded {ok}, skipped {skipped}, failed {failed}")
    if not with_photo and images_dir is None:
        print("no --images given, so nothing was background-stripped; add photos "
              "named <SKU>.jpg and re-run to attach them")
    if errors:
        print("\nerrors:")
        for line in errors[:20]:
            print(f"  {line}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-load the product catalogue.",
        epilog="Photos are matched by filename: <SKU>.jpg in --images.",
    )
    parser.add_argument("--api", help="Base URL of a running server. Required "
                                      "for background stripping.")
    parser.add_argument("--images", type=Path, help="Directory of product "
                                                    "photos named <SKU>.<ext>")
    parser.add_argument("--only", help="Load a single object_class (e.g. sofa)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Leave products already in the catalogue untouched")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would load, change nothing")
    parser.add_argument("--list", action="store_true",
                        help="Print the per-class counts and exit")
    args = parser.parse_args(argv)

    if args.list:
        counts = counts_by_class()
        for cls, n in sorted(counts.items()):
            print(f"{cls:15} {n}")
        print(f"{'TOTAL':15} {sum(counts.values())}")
        return 0

    if args.images and not args.images.is_dir():
        print(f"! --images {args.images} is not a directory", file=sys.stderr)
        return 2
    if args.images and not args.api:
        print("! --images needs --api: background stripping happens in the "
              "server, not here", file=sys.stderr)
        return 2

    if args.api:
        return seed_via_api(
            args.api.rstrip("/"),
            images_dir=args.images,
            dry_run=args.dry_run,
            only_class=args.only,
            skip_existing=args.skip_existing,
        )
    seed_direct(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())