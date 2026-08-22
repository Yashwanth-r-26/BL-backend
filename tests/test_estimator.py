"""Dimension priors, room classification, and estimation.

These cover the estimate path -- the honest stand-in for real measurement. The
throughline of every test: an estimate must be usable but never mistakable for
a fact. Confidence degrades when the model is unsure, and the source flag never
reads ``measured`` unless a real measurement set it.
"""

from __future__ import annotations

import pytest

from interior_ai.perception.estimator import (
    DIMENSION_SOURCE_ESTIMATED,
    DIMENSION_SOURCE_MEASURED,
    RoomClassification,
    build_estimated_room,
    estimate_dimensions,
    estimate_to_polygon,
    parse_classification,
)
from interior_ai.perception.priors import (
    HousingType,
    Region,
    RoomType,
    SizeBucket,
    lookup,
)


class TestPriors:
    def test_metro_living_average_is_realistic(self):
        dims, conf, _ = lookup(
            region=Region.IN_METRO, housing=HousingType.FLAT_2BHK,
            room=RoomType.LIVING, bucket=SizeBucket.AVERAGE,
        )
        # A Bangalore 2BHK living room ~16 m^2 is the documented typical.
        assert 12.0 <= dims.area_m2 <= 20.0

    def test_indian_ceilings_are_taller_than_western_default(self):
        dims, _, _ = lookup(
            region=Region.IN_METRO, housing=HousingType.FLAT_2BHK,
            room=RoomType.LIVING, bucket=SizeBucket.AVERAGE,
        )
        assert dims.ceiling_mm >= 2900

    def test_size_buckets_are_ordered(self):
        areas = [
            lookup(region=Region.IN_METRO, housing=HousingType.FLAT_2BHK,
                   room=RoomType.LIVING, bucket=b)[0].area_m2
            for b in (SizeBucket.SMALL, SizeBucket.AVERAGE, SizeBucket.LARGE)
        ]
        assert areas[0] < areas[1] < areas[2]

    def test_kitchen_smaller_than_living(self):
        living = lookup(region=Region.IN_METRO, housing=HousingType.FLAT_2BHK,
                        room=RoomType.LIVING, bucket=SizeBucket.AVERAGE)[0]
        kitchen = lookup(region=Region.IN_METRO, housing=HousingType.FLAT_2BHK,
                         room=RoomType.KITCHEN, bucket=SizeBucket.AVERAGE)[0]
        assert kitchen.area_m2 < living.area_m2

    def test_independent_house_scales_up(self):
        flat = lookup(region=Region.IN_METRO, housing=HousingType.FLAT_2BHK,
                      room=RoomType.LIVING, bucket=SizeBucket.AVERAGE)[0]
        house = lookup(region=Region.IN_METRO, housing=HousingType.INDEPENDENT,
                       room=RoomType.LIVING, bucket=SizeBucket.AVERAGE)[0]
        assert house.area_m2 > flat.area_m2

    def test_generic_fallback_has_lower_confidence(self):
        _, specific_conf, _ = lookup(
            region=Region.IN_METRO, housing=HousingType.FLAT_2BHK,
            room=RoomType.LIVING, bucket=SizeBucket.AVERAGE,
        )
        _, generic_conf, _ = lookup(
            region=Region.GENERIC, housing=HousingType.UNKNOWN,
            room=RoomType.LIVING, bucket=SizeBucket.AVERAGE,
        )
        assert generic_conf < specific_conf

    def test_dimensions_are_integer_mm(self):
        dims, _, _ = lookup(
            region=Region.IN_METRO, housing=HousingType.INDEPENDENT,
            room=RoomType.LIVING, bucket=SizeBucket.LARGE,
        )
        assert isinstance(dims.width_mm, int)
        assert isinstance(dims.depth_mm, int)


class TestClassificationParsing:
    def test_clean_json(self):
        c = parse_classification('{"room_type":"kitchen","size_class":"small"}')
        assert c.room_type is RoomType.KITCHEN
        assert c.size_bucket is SizeBucket.SMALL
        assert c.confidence > 0.7

    def test_fenced_json(self):
        c = parse_classification('```json\n{"room_type":"bedroom","size_class":"large"}\n```')
        assert c.room_type is RoomType.BEDROOM

    def test_unknown_room_type_lowers_confidence(self):
        c = parse_classification('{"room_type":"drawing_room","size_class":"average"}')
        assert c.room_type is RoomType.UNKNOWN
        assert c.confidence < 0.5

    def test_unknown_size_defaults_average_but_flags(self):
        c = parse_classification('{"room_type":"living","size_class":"unknown"}')
        assert c.size_bucket is SizeBucket.AVERAGE
        assert c.confidence < 0.8  # penalised for the unrecognised size

    def test_prose_degrades_to_unknown_low_confidence(self):
        c = parse_classification("This looks like a nice big living room to me")
        assert c.room_type is RoomType.UNKNOWN
        assert c.confidence <= 0.2

    def test_model_never_forced_to_emit_dimensions(self):
        """The classifier only ever yields a bucket, never a number -- there is
        no code path that turns model text into a measured dimension."""
        c = parse_classification('{"room_type":"living","size_class":"average"}')
        assert not hasattr(c, "width_mm")


class TestEstimation:
    def test_source_is_always_estimated(self):
        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        assert est.source == DIMENSION_SOURCE_ESTIMATED
        assert not est.is_measured

    def test_never_silently_becomes_measured(self):
        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 1.0)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        assert est.source != DIMENSION_SOURCE_MEASURED

    def test_confidence_compounds_classification_and_prior(self):
        """A shaky classification on a generic fallback must not yield a
        confident number."""
        weak = RoomClassification(RoomType.UNKNOWN, SizeBucket.AVERAGE, 0.2)
        est = estimate_dimensions(weak, region=Region.GENERIC, housing=HousingType.UNKNOWN)
        assert est.confidence < 0.15

    def test_strong_classification_yields_higher_confidence(self):
        strong = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        weak = RoomClassification(RoomType.UNKNOWN, SizeBucket.AVERAGE, 0.2)
        e_strong = estimate_dimensions(strong, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        e_weak = estimate_dimensions(weak, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        assert e_strong.confidence > e_weak.confidence

    def test_carries_a_caveat(self):
        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        assert "ESTIMATED" in est.caveat
        assert "not measured" in est.caveat.lower()

    def test_basis_records_the_reasoning(self):
        c = RoomClassification(RoomType.KITCHEN, SizeBucket.SMALL, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        assert "kitchen" in est.basis
        assert "small" in est.basis


class TestEstimatedRoom:
    def test_polygon_matches_dimensions(self):
        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        poly = estimate_to_polygon(est)
        assert poly[2].x == est.dimensions.width_mm
        assert poly[2].y == est.dimensions.depth_mm

    def test_room_is_valid_and_labelled_estimated(self):
        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        room = build_estimated_room(est)
        assert room.ceiling_height_mm == est.dimensions.ceiling_mm
        assert "estimated" in room.name.lower()

    def test_estimated_room_feeds_the_pipeline(self):
        """The estimate must produce a room the rest of the system accepts --
        that is the whole point of matching the scene contract."""
        from interior_ai.core.geometry import floor_area_mm2

        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        room = build_estimated_room(est)
        assert floor_area_mm2(room) > 0


class TestOpeningsInEstimate:
    def test_living_room_gets_door_and_window(self):
        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        room = build_estimated_room(est)
        kinds = {o.kind.value for o in room.openings}
        assert "door" in kinds
        assert "window" in kinds

    def test_openings_reduce_net_wall_area(self):
        from interior_ai.core.geometry import gross_wall_area_mm2, net_wall_area_mm2

        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        room = build_estimated_room(est)
        assert net_wall_area_mm2(room) < gross_wall_area_mm2(room)

    def test_openings_can_be_disabled(self):
        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        room = build_estimated_room(est, include_openings=False)
        assert room.openings == ()

    def test_paint_takeoff_deducts_openings(self):
        from interior_ai.pricing.takeoff import paint_takeoff

        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        with_op = build_estimated_room(est, include_openings=True)
        without = build_estimated_room(est, include_openings=False)
        paint_with = next(l for l in paint_takeoff(with_op) if l.sku == "PAINT-STD")
        paint_without = next(l for l in paint_takeoff(without) if l.sku == "PAINT-STD")
        assert float(paint_with.quantity) < float(paint_without.quantity)

    def test_estimated_door_has_a_swing(self):
        from interior_ai.core.enums import OpeningKind

        c = RoomClassification(RoomType.LIVING, SizeBucket.AVERAGE, 0.8)
        est = estimate_dimensions(c, region=Region.IN_METRO, housing=HousingType.FLAT_2BHK)
        room = build_estimated_room(est)
        door = next(o for o in room.openings if o.kind is OpeningKind.DOOR)
        assert door.swing is not None