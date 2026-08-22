"""Designer-grade top-down floor-plan rendering.

Turns a room and its solved placements into a floor plan that reads like an
interior designer drew it -- recognisable furniture shapes, architectural
walls, dimension lines, a legend and title block -- while every element sits at
the exact coordinate and rotation the CP-SAT solver produced.

This is deliberately *drawn from coordinates*, not generated. An image model
cannot honour precise geometry: give it "sofa at x=850" and it paints a sofa
somewhere plausible, not there. The whole value of a client floor plan is that
the sofa is where the sofa will actually go, so the beauty here comes from
better drawing code, not from generation. It stays a pure view of the scene --
it reads the graph and draws it, holding no state of its own.

Coordinates are the scene's own millimetres, scaled to fit the canvas, Y-axis
flipped so the plan reads the way someone standing at the door would see it.
Each furniture icon is drawn in the object's *local* frame and then rotated and
translated into place, so a rotated sofa's cushions rotate with it.
"""

from __future__ import annotations

import math

from ..core.enums import OpeningKind
from ..core.geometry import all_door_swings
from ..core.scene import Placement, Room

_WALL = "#2B2B2B"
_WALL_FILL = "#3A3A3A"
_FLOOR = "#F7F4EF"
_FLOOR_GRID = "#ECE7DE"
_DOOR = "#B5642F"
_WINDOW = "#5B8DB5"
_SWING = "#000000"
_DIM = "#8A8A8A"
_LABEL = "#2B2B2B"
_SUBTLE = "#6E6E6E"

_FURNITURE = {
    "sofa":         ("#9FB3CE", "#6E86A6"),
    "coffee_table": ("#D2B48C", "#A88A5F"),
    "tv_unit":      ("#8FB0A0", "#5F8271"),
    "armchair":     ("#B3C0D0", "#8595A8"),
    "bed":          ("#C7B8DC", "#9E8ABF"),
    "wardrobe":     ("#B0A08C", "#8A7860"),
    "side_table":   ("#D8C8A8", "#B0A078"),
    "dining_table": ("#D2B48C", "#A88A5F"),
    "bookshelf":    ("#A8A890", "#83835F"),
    "rug":          ("#E2D5C3", "#C4B39A"),
}
_DEFAULT_FURN = ("#BFBFBF", "#8F8F8F")

WALL_THICKNESS_PX = 8


def _fit(bounds, canvas, margin):
    minx, miny, maxx, maxy = bounds
    room_w = max(1, maxx - minx)
    room_h = max(1, maxy - miny)
    usable = canvas - 2 * margin
    scale = min(usable / room_w, usable / room_h)
    w_px = int(room_w * scale + 2 * margin)
    h_px = int(room_h * scale + 2 * margin)
    return scale, minx, miny, maxx, maxy, w_px, h_px


def _sofa(w, h, fill, stroke):
    arm = min(w, h) * 0.16
    back = h * 0.22
    parts = [
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="{h*0.08:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        f'<rect x="0" y="0" width="{w:.1f}" height="{back:.1f}" rx="{back*0.3:.1f}" fill="{stroke}" opacity="0.35"/>',
        f'<rect x="0" y="{back:.1f}" width="{arm:.1f}" height="{h-back:.1f}" fill="{stroke}" opacity="0.3"/>',
        f'<rect x="{w-arm:.1f}" y="{back:.1f}" width="{arm:.1f}" height="{h-back:.1f}" fill="{stroke}" opacity="0.3"/>',
    ]
    seats = 3 if w > h else 2
    inner_w = w - 2 * arm
    for i in range(1, seats):
        x = arm + inner_w * i / seats
        parts.append(f'<line x1="{x:.1f}" y1="{back:.1f}" x2="{x:.1f}" y2="{h:.1f}" stroke="{stroke}" stroke-width="1" opacity="0.4"/>')
    return "".join(parts)


def _coffee_table(w, h, fill, stroke):
    return (
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="{min(w,h)*0.12:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f'<rect x="{w*0.12:.1f}" y="{h*0.12:.1f}" width="{w*0.76:.1f}" height="{h*0.76:.1f}" rx="{min(w,h)*0.08:.1f}" fill="none" stroke="{stroke}" stroke-width="1" opacity="0.4"/>'
    )


def _tv_unit(w, h, fill, stroke):
    parts = [f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    long_side = max(w, h)
    divs = max(2, int(long_side / 40))
    for i in range(1, divs):
        if w >= h:
            x = w * i / divs
            parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h:.1f}" stroke="{stroke}" stroke-width="1" opacity="0.4"/>')
        else:
            y = h * i / divs
            parts.append(f'<line x1="0" y1="{y:.1f}" x2="{w:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1" opacity="0.4"/>')
    return "".join(parts)


def _armchair(w, h, fill, stroke):
    back = h * 0.24
    arm = w * 0.16
    return (
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="{min(w,h)*0.16:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f'<rect x="0" y="0" width="{w:.1f}" height="{back:.1f}" rx="{back*0.3:.1f}" fill="{stroke}" opacity="0.35"/>'
        f'<rect x="0" y="{back:.1f}" width="{arm:.1f}" height="{h-back:.1f}" fill="{stroke}" opacity="0.3"/>'
        f'<rect x="{w-arm:.1f}" y="{back:.1f}" width="{arm:.1f}" height="{h-back:.1f}" fill="{stroke}" opacity="0.3"/>'
    )


def _bed(w, h, fill, stroke):
    head = h * 0.12
    parts = [
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="{min(w,h)*0.04:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        f'<rect x="0" y="0" width="{w:.1f}" height="{head:.1f}" fill="{stroke}" opacity="0.4"/>',
    ]
    pw = w * 0.4
    ph = h * 0.18
    gap = w * 0.06
    for i in (0, 1):
        px = w / 2 - pw - gap / 2 + i * (pw + gap)
        parts.append(f'<rect x="{px:.1f}" y="{head+ph*0.3:.1f}" width="{pw:.1f}" height="{ph:.1f}" rx="{ph*0.4:.1f}" fill="white" stroke="{stroke}" stroke-width="1" opacity="0.85"/>')
    parts.append(f'<line x1="0" y1="{h*0.42:.1f}" x2="{w:.1f}" y2="{h*0.42:.1f}" stroke="{stroke}" stroke-width="1" opacity="0.4"/>')
    return "".join(parts)


def _wardrobe(w, h, fill, stroke):
    parts = [f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    doors = max(2, int(max(w, h) / 45))
    for i in range(1, doors):
        if w >= h:
            x = w * i / doors
            parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h:.1f}" stroke="{stroke}" stroke-width="1" opacity="0.5"/>')
        else:
            y = h * i / doors
            parts.append(f'<line x1="0" y1="{y:.1f}" x2="{w:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1" opacity="0.5"/>')
    return "".join(parts)


def _dining_table(w, h, fill, stroke):
    parts = [f'<rect x="{w*0.12:.1f}" y="{h*0.12:.1f}" width="{w*0.76:.1f}" height="{h*0.76:.1f}" rx="{min(w,h)*0.06:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    chair = min(w, h) * 0.14
    n = max(2, int(max(w, h) / 55))
    for i in range(n):
        t = (i + 0.5) / n
        if w >= h:
            cx = w * t - chair / 2
            parts.append(f'<rect x="{cx:.1f}" y="0" width="{chair:.1f}" height="{chair:.1f}" rx="2" fill="{stroke}" opacity="0.4"/>')
            parts.append(f'<rect x="{cx:.1f}" y="{h-chair:.1f}" width="{chair:.1f}" height="{chair:.1f}" rx="2" fill="{stroke}" opacity="0.4"/>')
        else:
            cy = h * t - chair / 2
            parts.append(f'<rect x="0" y="{cy:.1f}" width="{chair:.1f}" height="{chair:.1f}" rx="2" fill="{stroke}" opacity="0.4"/>')
            parts.append(f'<rect x="{w-chair:.1f}" y="{cy:.1f}" width="{chair:.1f}" height="{chair:.1f}" rx="2" fill="{stroke}" opacity="0.4"/>')
    return "".join(parts)


def _bookshelf(w, h, fill, stroke):
    parts = [f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    shelves = max(2, int(max(w, h) / 40))
    for i in range(1, shelves):
        if w >= h:
            x = w * i / shelves
            parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h:.1f}" stroke="{stroke}" stroke-width="1" opacity="0.5"/>')
        else:
            y = h * i / shelves
            parts.append(f'<line x1="0" y1="{y:.1f}" x2="{w:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1" opacity="0.5"/>')
    return "".join(parts)


def _generic(w, h, fill, stroke):
    return f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'


_ICONS = {
    "sofa": _sofa, "coffee_table": _coffee_table, "tv_unit": _tv_unit,
    "armchair": _armchair, "bed": _bed, "wardrobe": _wardrobe,
    "side_table": _generic, "dining_table": _dining_table,
    "bookshelf": _bookshelf, "rug": _generic,
}


def _draw_placement(p, scale, minx, room_h_mm, miny, margin):
    cls = p.object_class.value
    fill, stroke = _FURNITURE.get(cls, _DEFAULT_FURN)
    icon = _ICONS.get(cls, _generic)
    lw = p.footprint.width_mm * scale
    lh = p.footprint.depth_mm * scale
    minx_b, miny_b, maxx_b, maxy_b = p.bounds

    def sx(x):
        return (x - minx) * scale + margin

    def sy(y):
        return (room_h_mm - (y - miny)) * scale + margin

    yaw = p.yaw
    if yaw == 0:
        tx, ty, rot = sx(minx_b), sy(maxy_b), 0
    elif yaw == 90:
        tx, ty, rot = sx(maxx_b), sy(maxy_b), 90
    elif yaw == 180:
        tx, ty, rot = sx(maxx_b), sy(miny_b), 180
    else:
        tx, ty, rot = sx(minx_b), sy(miny_b), 270
    body = icon(lw, lh, fill, stroke)
    return f'<g transform="translate({tx:.1f},{ty:.1f}) rotate({rot})">{body}</g>'


def _dimension_line(x1, y1, x2, y2, label, colour):
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1
    px, py = -dy / length, dx / length
    tick = 5
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    lox, loy = px * 10, py * 10
    rot = math.degrees(math.atan2(dy, dx))
    if rot > 90 or rot < -90:
        rot += 180
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" stroke-width="1"/>'
        f'<line x1="{x1-px*tick:.1f}" y1="{y1-py*tick:.1f}" x2="{x1+px*tick:.1f}" y2="{y1+py*tick:.1f}" stroke="{colour}" stroke-width="1"/>'
        f'<line x1="{x2-px*tick:.1f}" y1="{y2-py*tick:.1f}" x2="{x2+px*tick:.1f}" y2="{y2+py*tick:.1f}" stroke="{colour}" stroke-width="1"/>'
        f'<text x="{mid_x+lox:.1f}" y="{mid_y+loy:.1f}" font-size="11" fill="{colour}" text-anchor="middle" dominant-baseline="middle" transform="rotate({rot:.1f} {mid_x+lox:.1f} {mid_y+loy:.1f})">{label}</text>'
    )


def render_floor_plan(room, *, canvas=900, margin=70, show_swings=True,
                      show_labels=True, show_dimensions=True, show_legend=True,
                      title=None, subtitle=None):
    """Render a room and its placements as a designer-grade top-down SVG."""
    scale, minx, miny, maxx, maxy, w_px, h_px = _fit(room.bounds, canvas, margin)
    room_h_mm = maxy - miny
    room_w_mm = maxx - minx

    def sx(x):
        return (x - minx) * scale + margin

    def sy(y):
        return (room_h_mm - (y - miny)) * scale + margin

    P = []
    P.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_px}" height="{h_px}" viewBox="0 0 {w_px} {h_px}" font-family="system-ui,-apple-system,sans-serif">')
    P.append(f'<rect width="{w_px}" height="{h_px}" fill="white"/>')
    P.append(f'<defs><filter id="sh" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.18"/></filter></defs>')

    poly = " ".join(f"{sx(p.x):.1f},{sy(p.y):.1f}" for p in room.polygon)
    P.append(f'<polygon points="{poly}" fill="{_WALL_FILL}" stroke="{_WALL}" stroke-width="{WALL_THICKNESS_PX*2}" stroke-linejoin="round" filter="url(#sh)"/>')
    P.append(f'<polygon points="{poly}" fill="{_FLOOR}" stroke="none"/>')

    step_mm = 1000
    gx = (minx // step_mm + 1) * step_mm
    while gx < maxx:
        P.append(f'<line x1="{sx(gx):.1f}" y1="{sy(miny):.1f}" x2="{sx(gx):.1f}" y2="{sy(maxy):.1f}" stroke="{_FLOOR_GRID}" stroke-width="1"/>')
        gx += step_mm
    gy = (miny // step_mm + 1) * step_mm
    while gy < maxy:
        P.append(f'<line x1="{sx(minx):.1f}" y1="{sy(gy):.1f}" x2="{sx(maxx):.1f}" y2="{sy(gy):.1f}" stroke="{_FLOOR_GRID}" stroke-width="1"/>')
        gy += step_mm

    if show_swings:
        for sw in all_door_swings(room):
            if sw.is_empty:
                continue
            coords = list(sw.exterior.coords)
            sp = " ".join(f"{sx(int(x)):.1f},{sy(int(y)):.1f}" for x, y in coords)
            P.append(f'<polygon points="{sp}" fill="none" stroke="{_SWING}" stroke-width="1" stroke-dasharray="4 3" opacity="0.4"/>')

    for op in room.openings:
        cx, cy = op.centre.x, op.centre.y
        half = op.width_mm // 2
        if op.wall_index in (0, 2):
            x1, x2 = sx(cx - half), sx(cx + half)
            y1 = y2 = sy(cy)
        else:
            x1 = x2 = sx(cx)
            y1, y2 = sy(cy - half), sy(cy + half)
        if op.kind is OpeningKind.DOOR:
            P.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{_FLOOR}" stroke-width="{WALL_THICKNESS_PX*2}"/>')
            P.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{_DOOR}" stroke-width="3"/>')
        else:
            P.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{_FLOOR}" stroke-width="{WALL_THICKNESS_PX*2}"/>')
            P.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{_WINDOW}" stroke-width="4"/>')
            P.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="white" stroke-width="1"/>')

    for p in room.placements:
        P.append(_draw_placement(p, scale, minx, room_h_mm, miny, margin))
        if show_labels:
            b = p.bounds
            lx = sx((b[0] + b[2]) // 2)
            ly = sy((b[1] + b[3]) // 2)
            label = p.object_class.value.replace("_", " ")
            P.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10.5" fill="{_LABEL}" text-anchor="middle" dominant-baseline="middle" opacity="0.85" paint-order="stroke" stroke="white" stroke-width="2.5">{label}</text>')

    if show_dimensions:
        off = 24
        bx1, bx2 = sx(minx), sx(maxx)
        by = sy(miny) + off
        P.append(_dimension_line(bx1, by, bx2, by, f"{room_w_mm/1000:.2f} m", _DIM))
        ly1, ly2 = sy(miny), sy(maxy)
        lx = sx(minx) - off
        P.append(_dimension_line(lx, ly1, lx, ly2, f"{room_h_mm/1000:.2f} m", _DIM))

    ttl = title or room.name
    P.append(f'<text x="{margin}" y="30" font-size="18" font-weight="700" fill="{_LABEL}">{ttl}</text>')
    area = (room_w_mm * room_h_mm) / 1_000_000
    sub = subtitle or f"{area:.1f} m\u00b2  \u00b7  {room_w_mm/1000:.2f} \u00d7 {room_h_mm/1000:.2f} m  \u00b7  ceiling {room.ceiling_height_mm/1000:.2f} m"
    P.append(f'<text x="{margin}" y="48" font-size="11.5" fill="{_SUBTLE}">{sub}</text>')

    if show_legend and room.placements:
        classes = []
        for p in room.placements:
            c = p.object_class.value
            if c not in classes:
                classes.append(c)
        lx = w_px - margin - 130
        ly = 24
        P.append(f'<rect x="{lx-10:.1f}" y="{ly-14:.1f}" width="140" height="{18*len(classes)+16:.1f}" rx="6" fill="white" stroke="{_FLOOR_GRID}" stroke-width="1" opacity="0.95"/>')
        for i, c in enumerate(classes):
            fill, stroke = _FURNITURE.get(c, _DEFAULT_FURN)
            yy = ly + i * 18
            P.append(f'<rect x="{lx:.1f}" y="{yy:.1f}" width="12" height="12" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
            P.append(f'<text x="{lx+18:.1f}" y="{yy+10:.1f}" font-size="10.5" fill="{_LABEL}">{c.replace("_"," ")}</text>')

    nx, ny = w_px - margin + 4, h_px - margin - 10
    P.append(f'<g transform="translate({nx:.1f},{ny:.1f})"><line x1="0" y1="10" x2="0" y2="-10" stroke="{_SUBTLE}" stroke-width="1.5"/><path d="M0,-12 L4,-4 L-4,-4 Z" fill="{_SUBTLE}"/><text x="0" y="22" font-size="9" fill="{_SUBTLE}" text-anchor="middle">N</text></g>')

    P.append("</svg>")
    return "".join(P)