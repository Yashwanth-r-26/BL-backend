"""Takeoff arithmetic and price snapshotting."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from interior_ai.core.enums import ObjectClass, PriceStatus, Unit
from interior_ai.core.geometry import net_wall_area_mm2
from interior_ai.core.scene import Footprint, Placement, Scene, Vec2
from interior_ai.core.units import mm2_to_m2
from interior_ai.pricing.engine import PricingEngine
from interior_ai.pricing.prices import PriceBook, PriceObservation
from interior_ai.pricing.takeoff import (
    TILE_WASTAGE_PCT,
    flooring_takeoff,
    furniture_takeoff,
    paint_takeoff,
    room_takeoff,
)


class TestFlooringTakeoff:
    def test_tile_quantity_includes_wastage(self, windowed_room):
        lines = {l.sku: l for l in flooring_takeoff(windowed_room)}
        floor_m2 = Decimal("12.00")
        expected = floor_m2 * Decimal(str(1 + TILE_WASTAGE_PCT))
        assert float(lines["TILE-STD"].quantity) == pytest.approx(float(expected), rel=1e-3)

    def test_wastage_is_not_optional(self, windowed_room):
        """Skipping wastage is the classic underquote -- tiles are cut at every
        edge and a few always break."""
        line = next(l for l in flooring_takeoff(windowed_room) if l.sku == "TILE-STD")
        assert float(line.quantity) > 12.0

    def test_adhesive_scales_with_laid_area(self, windowed_room):
        lines = {l.sku: l for l in flooring_takeoff(windowed_room)}
        tiles = float(lines["TILE-STD"].quantity)
        assert float(lines["ADHESIVE-STD"].quantity) == pytest.approx(tiles * 4.0, rel=1e-3)

    def test_every_line_shows_its_arithmetic(self, windowed_room):
        for line in flooring_takeoff(windowed_room):
            assert line.basis


class TestPaintTakeoff:
    def test_uses_net_not_gross_wall_area(self, windowed_room):
        """Quoting the gross area bills the client for painting their windows."""
        net_m2 = mm2_to_m2(net_wall_area_mm2(windowed_room))
        paint = next(l for l in paint_takeoff(windowed_room) if l.sku == "PAINT-STD")
        expected = net_m2 * 2 / 9.0
        assert float(paint.quantity) == pytest.approx(expected, rel=1e-3)

    def test_openings_reduce_the_quantity(self, windowed_room, bare_room):
        """Same-size room without openings must need more paint."""
        with_openings = next(
            l for l in paint_takeoff(windowed_room) if l.sku == "PAINT-STD"
        )
        without = next(l for l in paint_takeoff(bare_room) if l.sku == "PAINT-STD")
        assert float(without.quantity) > float(with_openings.quantity)

    def test_basis_names_the_deduction(self, windowed_room):
        paint = next(l for l in paint_takeoff(windowed_room) if l.sku == "PAINT-STD")
        assert "openings" in paint.basis

    def test_materials_have_independent_coverage_rates(self, windowed_room):
        lines = {l.sku: l for l in paint_takeoff(windowed_room)}
        assert lines["PUTTY-STD"].unit is Unit.KG
        assert lines["PRIMER-STD"].unit is Unit.LITRE
        assert float(lines["PRIMER-STD"].quantity) != float(lines["PAINT-STD"].quantity)


class TestFurnitureTakeoff:
    def test_counts_placements(self, bare_room):
        fp = Footprint(width_mm=800, depth_mm=800, height_mm=900)
        room = bare_room.model_copy(
            update={
                "placements": (
                    Placement(sku="AC-1", object_class=ObjectClass.ARMCHAIR,
                              origin=Vec2(x=0, y=0), footprint=fp),
                    Placement(sku="AC-1", object_class=ObjectClass.ARMCHAIR,
                              origin=Vec2(x=1000, y=0), footprint=fp),
                )
            }
        )
        lines = furniture_takeoff(room)
        assert len(lines) == 1
        assert lines[0].quantity == Decimal(2)

    def test_obstacles_are_not_priced(self, obstacle_room):
        """Obstacles are existing building fabric, not something being bought."""
        assert furniture_takeoff(obstacle_room) == []


class TestPriceHistory:
    def test_history_is_append_only(self, now):
        book = PriceBook()
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("100"), observed_at=now - timedelta(days=5)))
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("120"), observed_at=now))
        assert len(book.history_for("X")) == 2

    def test_current_is_the_latest_observation(self, now):
        book = PriceBook()
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("100"), observed_at=now - timedelta(days=5)))
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("120"), observed_at=now))
        assert book.current("X").amount == Decimal("120")

    def test_backfilled_observation_does_not_clobber_newer(self, now):
        """Inserting an old observation later is legitimate and must not
        overwrite a newer price just because it arrived after it."""
        book = PriceBook()
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("120"), observed_at=now))
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("80"), observed_at=now - timedelta(days=90)))
        assert book.current("X").amount == Decimal("120")

    def test_negative_prices_rejected(self):
        with pytest.raises(ValueError):
            PriceObservation(sku="X", vendor="V", unit=Unit.SQM, amount=Decimal("-1"))


class TestSnapshots:
    def test_fresh_price(self, price_book):
        assert price_book.snapshot("TILE-STD").status is PriceStatus.FRESH

    def test_stale_price_is_flagged_not_hidden(self, price_book):
        """An old price is still the best information available; suppressing it
        leaves a hole in the quote."""
        snap = price_book.snapshot("PAINT-STD")
        assert snap.status is PriceStatus.STALE
        assert snap.amount is not None
        assert snap.age_days >= 7

    def test_unpriced_sku_surfaces(self, price_book):
        snap = price_book.snapshot("NONEXISTENT")
        assert snap.status is PriceStatus.UNPRICED
        assert snap.amount is None
        assert not snap.is_usable

    def test_snapshot_carries_vendor_and_date(self, price_book):
        snap = price_book.snapshot("TILE-STD")
        assert snap.vendor == "Kajaria"
        assert snap.observed_at is not None

    def test_snapshot_explains_itself(self, price_book):
        assert "STALE" in price_book.snapshot("PAINT-STD").explain()
        assert "no price" in price_book.snapshot("NONEXISTENT").explain()


class TestQuoting:
    def test_unpriced_lines_excluded_from_total(self, windowed_room, price_book):
        """A missing price that silently becomes zero produces a confident
        underquote nobody notices until the invoice arrives."""
        scene = Scene(rooms=(windowed_room,))
        quote = PricingEngine(price_book).quote_room(scene, windowed_room, include_furniture=False)
        assert quote.unpriced_lines
        assert not quote.is_complete
        for line in quote.unpriced_lines:
            assert line.line_total is None

    def test_unpriced_skus_named_in_warnings(self, windowed_room, price_book):
        scene = Scene(rooms=(windowed_room,))
        quote = PricingEngine(price_book).quote_room(scene, windowed_room, include_furniture=False)
        joined = " ".join(quote.warnings())
        assert "PUTTY-STD" in joined

    def test_stale_lines_counted_but_reported(self, windowed_room, price_book):
        scene = Scene(rooms=(windowed_room,))
        quote = PricingEngine(price_book).quote_room(scene, windowed_room, include_furniture=False)
        assert quote.stale_total > 0
        assert quote.stale_total <= quote.total
        assert any("older than" in w for w in quote.warnings())

    def test_total_is_sum_of_priced_lines(self, windowed_room, price_book):
        scene = Scene(rooms=(windowed_room,))
        quote = PricingEngine(price_book).quote_room(scene, windowed_room, include_furniture=False)
        expected = sum(l.line_total for l in quote.priced_lines)
        assert quote.total == expected

    def test_quote_pins_the_scene_version(self, windowed_room, price_book):
        """The field that makes a six-month-old quote reproducible."""
        scene = Scene(rooms=(windowed_room,))
        quote = PricingEngine(price_book).quote_room(scene, windowed_room)
        assert quote.scene_version_id == scene.version_id

    def test_reprice_of_old_version_is_stable(self, windowed_room, price_book, now):
        """Recording a new price must not change what an existing quote says."""
        scene = Scene(rooms=(windowed_room,))
        engine = PricingEngine(price_book)
        first = engine.quote_room(scene, windowed_room, as_of=now, include_furniture=False)
        original_total = first.total

        price_book.record(
            PriceObservation(sku="TILE-STD", vendor="Kajaria", unit=Unit.SQM,
                             amount=Decimal("9999"), observed_at=now + timedelta(days=1))
        )
        assert first.total == original_total

    def test_quote_scene_prices_every_room(self, windowed_room, bare_room, price_book):
        scene = Scene(rooms=(windowed_room, bare_room))
        quote = PricingEngine(price_book).quote_scene(scene, include_furniture=False)
        room_ids = {l.room_id for l in quote.lines}
        assert windowed_room.id in room_ids
        assert bare_room.id in room_ids

    def test_line_totals_are_quantity_times_price(self, windowed_room, price_book):
        scene = Scene(rooms=(windowed_room,))
        quote = PricingEngine(price_book).quote_room(scene, windowed_room, include_furniture=False)
        tile = next(l for l in quote.lines if l.sku == "TILE-STD")
        assert tile.line_total == (tile.quantity * tile.price.amount).quantize(Decimal("0.01"))
