"""Shapely bridge for the scene graph.

Scene objects store integer millimetres; Shapely works in floats. Every
conversion happens here so the float representation stays confined to
geometric predicates and never leaks back into stored state.

Door swing arcs are modelled as quarter-circle sectors approximated by
polygons. A door is not a rectangle of blocked floor -- treating it as one
either over-blocks (a whole square of unusable room) or under-blocks (the leaf
sweeps through your sofa). The sector is the honest shape.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .enums import OpeningKind, SwingDirection
from .scene import Obstacle, Opening, Placement, Room, Vec2


def room_polygon(room: Room) -> Polygon:
    """Room floor as a Shapely polygon."""
    return Polygon([(p.x, p.y) for p in room.polygon])


def placement_polygon(placement: Placement) -> Polygon:
    """Axis-aligned box of a placed object."""
    minx, miny, maxx, maxy = placement.bounds
    return box(minx, miny, maxx, maxy)


def obstacle_polygon(obstacle: Obstacle) -> Polygon:
    minx, miny, maxx, maxy = obstacle.bounds
    return box(minx, miny, maxx, maxy)


def bounds_polygon(bounds: tuple[int, int, int, int]) -> Polygon:
    return box(*bounds)


def _wall_inward_normal(room: Room, wall_index: int) -> tuple[float, float]:
    """Inward-pointing unit normal of a wall segment.

    Computed from the polygon rather than assumed, so L-shaped rooms and
    reversed vertex winding both behave.
    """
    poly = room.polygon
    n = len(poly)
    a = poly[wall_index % n]
    b = poly[(wall_index + 1) % n]
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy)
    if length == 0:
        return (0.0, 0.0)
    # Two candidate normals; pick whichever points into the room.
    nx, ny = -dy / length, dx / length
    mid = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
    probe = (mid[0] + nx * 1.0, mid[1] + ny * 1.0)
    rp = room_polygon(room)
    from shapely.geometry import Point

    if rp.contains(Point(probe)):
        return (nx, ny)
    return (-nx, -ny)


def door_swing_polygon(room: Room, opening: Opening, segments: int = 12) -> Polygon | None:
    """Quarter-circle sector swept by a door leaf.

    Returns ``None`` for windows and for outward-swinging doors -- an outward
    door consumes floor in the *next* room, not this one, so blocking floor
    here would be wrong.
    """
    if opening.kind is not OpeningKind.DOOR:
        return None
    if opening.swing is SwingDirection.OUTWARD:
        return None

    radius = float(opening.swing_radius_mm or opening.width_mm)
    nx, ny = _wall_inward_normal(room, opening.wall_index)
    if nx == 0.0 and ny == 0.0:
        return None

    # Hinge sits at one edge of the opening, along the wall direction.
    poly = room.polygon
    n = len(poly)
    a = poly[opening.wall_index % n]
    b = poly[(opening.wall_index + 1) % n]
    wx, wy = b.x - a.x, b.y - a.y
    wlen = math.hypot(wx, wy) or 1.0
    ux, uy = wx / wlen, wy / wlen

    cx = opening.centre.x - ux * (opening.width_mm / 2.0)
    cy = opening.centre.y - uy * (opening.width_mm / 2.0)

    start = math.atan2(uy, ux)
    end = math.atan2(ny, nx)
    # Sweep the short way round from wall direction to inward normal.
    delta = (end - start + math.pi) % (2 * math.pi) - math.pi

    pts = [(cx, cy)]
    for i in range(segments + 1):
        t = start + delta * (i / segments)
        pts.append((cx + radius * math.cos(t), cy + radius * math.sin(t)))
    sector = Polygon(pts)
    if not sector.is_valid:
        sector = sector.buffer(0)
    return sector


def all_door_swings(room: Room) -> list[Polygon]:
    out = []
    for op in room.openings:
        sw = door_swing_polygon(room, op)
        if sw is not None and not sw.is_empty:
            out.append(sw)
    return out


def blocked_region(room: Room, *, include_placements: bool = True) -> Polygon:
    """Union of everything a new object may not overlap."""
    parts: list[Polygon] = [obstacle_polygon(o) for o in room.obstacles]
    parts.extend(all_door_swings(room))
    if include_placements:
        parts.extend(placement_polygon(p) for p in room.placements)
    if not parts:
        return Polygon()
    return unary_union(parts)


def front_clearance_polygon(placement: Placement, clearance_mm: int) -> Polygon | None:
    """Rectangle of space an object needs in front of it.

    Extends from the object's front face along its facing direction. Returns
    ``None`` when the item declares no clearance requirement.
    """
    if clearance_mm <= 0:
        return None
    minx, miny, maxx, maxy = placement.bounds
    fx, fy = placement.facing_vector()
    if (fx, fy) == (0, -1):
        return box(minx, miny - clearance_mm, maxx, miny)
    if (fx, fy) == (0, 1):
        return box(minx, maxy, maxx, maxy + clearance_mm)
    if (fx, fy) == (1, 0):
        return box(maxx, miny, maxx + clearance_mm, maxy)
    return box(minx - clearance_mm, miny, minx, maxy)


def wall_segments(room: Room) -> list[tuple[Vec2, Vec2]]:
    poly = room.polygon
    n = len(poly)
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]


def distance_to_nearest_wall(room: Room, placement: Placement) -> float:
    """Shortest distance from an object's box to any wall segment."""
    from shapely.geometry import LineString

    pp = placement_polygon(placement)
    best = float("inf")
    for a, b in wall_segments(room):
        seg = LineString([(a.x, a.y), (b.x, b.y)])
        best = min(best, pp.distance(seg))
    return best


def floor_area_mm2(room: Room) -> int:
    """Room floor area in square millimetres."""
    return int(round(room_polygon(room).area))


def perimeter_mm(room: Room) -> int:
    return int(round(room_polygon(room).length))


def opening_area_mm2(opening: Opening) -> int:
    return opening.width_mm * opening.height_mm


def gross_wall_area_mm2(room: Room) -> int:
    """Perimeter times ceiling height, before subtracting openings."""
    return perimeter_mm(room) * room.ceiling_height_mm


def net_wall_area_mm2(room: Room) -> int:
    """Paintable wall area: gross minus every opening.

    Clamped at zero -- a room whose openings exceed its walls is a data error,
    but returning a negative area would silently produce a negative paint
    quantity downstream, which is worse than returning nothing.
    """
    gross = gross_wall_area_mm2(room)
    holes = sum(opening_area_mm2(o) for o in room.openings)
    return max(0, gross - holes)
