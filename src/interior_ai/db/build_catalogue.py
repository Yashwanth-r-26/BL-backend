"""Build the whole catalogue in one pass.

For every product: generate a studio photograph on white from its own
specification, then store the product, that image and an opening price. One
command, one image-generation call each::

    python -m interior_ai.db.build_catalogue

Why this and not "generate to a folder, then upload": the generated image is
*already* isolated on white, so putting it through the upload endpoint's
background strip would spend a second image-generation call per product to
produce what it started with. At a hundred products that is hours of API time
bought for nothing. Images that arrive already isolated -- these, or a
vendor's cut-out shots -- skip the strip.

Resumable by design. Generation is slow and paid, so a product that already
has an image is left alone unless ``--overwrite`` says otherwise; an
interrupted run continues where it stopped.

Options::

    --only sofa          just one object class
    --limit 10           stop after N products
    --overwrite          regenerate images that already exist
    --no-images          specifications and prices only, no generation
    --save-dir DIR       also write each image to disk for inspection
    --dry-run            show the plan, call nothing

On accuracy: these are generated from the catalogue's own descriptions, so the
picker thumbnail and the object swapped into a customer's room are the same
thing. Real supplier photography is better still -- it is the actual object a
customer receives -- and can replace any of these through /admin without
re-entering the product.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import sys
import time
from decimal import Decimal
from pathlib import Path

from .product_catalogue import MATERIALS, PRODUCTS, TREATMENTS


def _decode(data_uri: str) -> bytes:
    _, _, b64 = data_uri.partition(",")
    return base64.b64decode(b64)


def build(
    *,
    only_class: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    with_images: bool = True,
    include_treatments: bool = True,
    save_dir: Path | None = None,
    dry_run: bool = False,
    pause_s: float = 0.0,
    max_attempts: int = 5,
    abort_after: int = 3,
) -> int:
    """Populate the catalogue. Returns a process exit code."""
    import os

    from ..perception.editing import GeminiPhotoEditor, MockPhotoEditor
    from ..providers.base import ProviderError
    from .catalogue import CatalogueItemRow
    from .repository import make_engine, make_session_factory

    products = [p for p in PRODUCTS if only_class in (None, p[2])]
    treatments = (
        [t for t in TREATMENTS if only_class in (None, t[2])]
        if include_treatments else []
    )
    if limit:
        products = products[:limit]

    have_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("CLOUD_API_KEY"))
    editor = GeminiPhotoEditor() if have_key else MockPhotoEditor()
    if have_key and max_attempts != 5:
        # Retry budget is per image; the editor reads it from the call site.
        editor.default_attempts = max_attempts
    database = os.getenv("DATABASE_URL")

    print(f"database    {database or 'NOT SET -- in-memory, nothing will persist'}")
    print(f"products    {len(products)}" + (f" (class {only_class})" if only_class else ""))
    print(f"treatments  {len(treatments)}")
    if with_images:
        print(f"generator   {type(editor).__name__}"
              + ("" if have_key else "  -- no GEMINI_API_KEY: placeholder cards, not product photos"))
    else:
        print("images      skipped (--no-images)")
    print()

    if not database:
        print("! DATABASE_URL is not set. Set it in .env first, or this run is "
              "written to a database that disappears when the process exits.",
              file=sys.stderr)
        return 2

    if dry_run:
        for sku, name, cls, *_ in products:
            print(f"  [dry-run] {cls:14} {sku:20} {name}")
        for sku, name, cls, *_ in treatments:
            print(f"  [dry-run] {cls:14} {sku:20} {name}")
        return 0

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    engine = make_engine()
    session_factory = make_session_factory(engine)

    stored = generated = skipped_image = failed = 0
    consecutive_failures = 0
    errors: list[str] = []
    started = time.time()

    def report_retry(model, attempt, attempts, delay, exc):
        if delay:
            reason = getattr(exc, "status_code", None) or "network"
            print(f"      retry {attempt}/{attempts} on {model} after {reason} "
                  f"-- waiting {delay:.0f}s", flush=True)
        else:
            print(f"      {model} exhausted, trying the next model", flush=True)

    for index, (sku, name, cls, w, d, h, price, desc) in enumerate(products, start=1):
        label = f"[{index}/{len(products)}] {sku}"
        image_uri: str | None = None

        if with_images:
            with session_factory() as db:
                existing = db.get(CatalogueItemRow, sku)
                has_image = bool(existing and (existing.image_ref or "").startswith("data:"))
            if has_image and not overwrite:
                skipped_image += 1
                print(f"{label} image already present, keeping it")
            else:
                try:
                    print(f"{label} generating image …", flush=True)
                    kwargs = {"name": name, "description": desc, "object_class": cls}
                    if "on_retry" in inspect.signature(
                        editor.generate_product_image
                    ).parameters:
                        kwargs["on_retry"] = report_retry
                    image_uri = editor.generate_product_image(**kwargs)
                    generated += 1
                    consecutive_failures = 0
                    if save_dir:
                        (save_dir / f"{sku}.png").write_bytes(_decode(image_uri))
                except ProviderError as exc:
                    failed += 1
                    consecutive_failures += 1
                    errors.append(f"{sku}: {exc}")
                    print(f"{label} image FAILED after retries: {exc}")
                    # If the model stays unavailable, continuing would store a
                    # hundred products without photographs and call it done.
                    # Stop while the run is still cheap to resume.
                    if abort_after and consecutive_failures >= abort_after:
                        print(f"\n{consecutive_failures} products failed in a row -- "
                              "the image model looks unavailable. Stopping here; "
                              "everything stored so far is kept, so re-running "
                              "continues from this point.")
                        if getattr(exc, "status_code", None) == 503:
                            print("503 means the model is busy, not that anything "
                                  "is wrong with the request. Wait a few minutes, "
                                  "or set GEMINI_IMAGE_FALLBACKS to a model with "
                                  "spare capacity.")
                        return 1
                except KeyboardInterrupt:
                    print("\ninterrupted -- everything stored so far is kept; "
                          "re-run to continue")
                    return 130

        try:
            with session_factory() as db:
                row = db.get(CatalogueItemRow, sku) or CatalogueItemRow(sku=sku)
                row.name, row.object_class, row.description = name, cls, desc
                row.width_mm, row.depth_mm, row.height_mm = w, d, h
                row.display_price = Decimal(price)
                row.currency, row.active = "INR", 1
                if image_uri:
                    row.image_ref = image_uri
                db.add(row)
                db.commit()
            stored += 1
            print(f"{label} stored" + (" with image" if image_uri else ""))
        except Exception as exc:
            failed += 1
            errors.append(f"{sku} (store): {exc}")
            print(f"{label} STORE FAILED: {exc}")

        if pause_s and index < len(products):
            time.sleep(pause_s)

    # Surface finishes: a swatch or material sample, no object photo needed.
    for sku, name, cls, price, desc, tags in treatments:
        try:
            with session_factory() as db:
                row = db.get(CatalogueItemRow, sku) or CatalogueItemRow(sku=sku)
                row.name, row.object_class, row.description = name, cls, desc
                row.width_mm = row.depth_mm = row.height_mm = 1
                row.display_price = Decimal(price)
                row.currency, row.active = "INR", 1
                row.style_tags = tags
                db.add(row)
                db.commit()
            stored += 1
        except Exception as exc:
            failed += 1
            errors.append(f"{sku}: {exc}")

    # Opening prices so quotes are complete on day one; real vendor prices
    # overwrite these through /prices.
    from ..core.enums import Unit
    from ..db.stores import SqlPriceBookAdapter
    from ..pricing.prices import PriceObservation

    book = SqlPriceBookAdapter(session_factory)
    observations = [
        PriceObservation(sku=sku, vendor="Catalogue (opening)", unit=Unit.PIECE,
                         amount=Decimal(price), source="build_catalogue")
        for sku, _n, _c, _w, _d, _h, price, _desc in products
    ] + [
        PriceObservation(sku=sku, vendor="Catalogue (opening)", unit=Unit.PIECE,
                         amount=Decimal(price), source="build_catalogue")
        for sku, _n, _c, price, _desc, _t in treatments
    ] + [
        PriceObservation(sku=sku, vendor="Local", unit=Unit(unit),
                         amount=Decimal(amount), source="build_catalogue")
        for sku, unit, amount in MATERIALS
    ]
    try:
        book.record_many(observations)
    except Exception as exc:
        errors.append(f"prices: {exc}")

    elapsed = time.time() - started
    print()
    print(f"stored {stored} items, generated {generated} images, "
          f"kept {skipped_image} existing, failed {failed} in {elapsed / 60:.1f} min")
    print(f"recorded {len(observations)} opening prices")
    if errors:
        print("\nerrors:")
        for line in errors[:20]:
            print(f"  {line}")
    if not failed:
        print("\ncatalogue ready -- open /admin to review, /ui to use it")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate product images and store the whole catalogue in "
                    "one pass.",
    )
    parser.add_argument("--only", help="Restrict to one object_class, e.g. sofa")
    parser.add_argument("--limit", type=int, help="Stop after N products")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate images for products that already have one")
    parser.add_argument("--no-images", action="store_true",
                        help="Specifications and prices only")
    parser.add_argument("--no-treatments", action="store_true",
                        help="Skip wall/ceiling/floor finishes")
    parser.add_argument("--save-dir", type=Path,
                        help="Also write generated images here for inspection")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Seconds between generations (rate limiting)")
    parser.add_argument("--attempts", type=int, default=5,
                        help="Retries per image before giving up (default 5)")
    parser.add_argument("--abort-after", type=int, default=3,
                        help="Stop after N consecutive image failures; 0 to "
                             "keep going regardless (default 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the plan, call nothing")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        print("! --limit must be positive", file=sys.stderr)
        return 2

    return build(
        only_class=args.only,
        limit=args.limit,
        overwrite=args.overwrite,
        with_images=not args.no_images,
        include_treatments=not args.no_treatments,
        save_dir=args.save_dir,
        dry_run=args.dry_run,
        pause_s=args.pause,
        max_attempts=args.attempts,
        abort_after=args.abort_after,
    )


if __name__ == "__main__":
    raise SystemExit(main())