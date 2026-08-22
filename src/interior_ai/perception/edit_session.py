"""Edit-session service -- the select/swap/iterate loop.

Orchestrates one user's photo-editing session end to end:

    start(photo)          -> detect objects once, store session
    select(click x,y)     -> hit-test to a detection, offer catalogue items
                             of that class (fit-checked against the room)
    apply(detection, sku) -> image-edit replacement, append a step
    undo / current        -> move the step pointer; steps are never deleted
    swapped_skus()        -> what the *current* image contains, for the quote

The step chain is append-only for the same reason scene versions are: the
final quote must be able to name exactly which swaps produced the image it
priced, and "undo" that deletes history would sever that trail.

Fit-gating the picker: a catalogue sofa that cannot physically fit the room is
filtered out *before* the user sees it, using the same fit engine the solver
trusts. Offering a product the room cannot take, letting the user fall in love
with it, and failing later is the worst order of operations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from ..core.enums import ObjectClass
from ..core.scene import CatalogueItem, Footprint, Room, Vec2
from ..db.catalogue import CatalogueItemRow, EditSession, EditStep
from ..fit.engine import FitEngine
from .editing import Detection, hit_test

# Detection classes that map onto solvable catalogue classes. Others (lamp,
# curtain, wall_art, plant) are still selectable/replaceable but not
# fit-checked -- a lamp does not meaningfully "not fit" a room.
_FIT_CHECKED = {c.value for c in ObjectClass}


@dataclass(frozen=True)
class Offer:
    """One catalogue item offered for a selected object."""

    sku: str
    name: str
    object_class: str
    width_mm: int
    depth_mm: int
    height_mm: int
    display_price: str
    currency: str
    image_ref: str | None
    fits_room: bool
    fit_note: str | None = None
    # From style_tags: {"suggested": true} floats an item to the top of the
    # picker; {"hex": "#AABBCC"} gives paint options a renderable swatch.
    suggested: bool = False
    swatch: str | None = None


class OversizeReplacement(Exception):
    """The chosen product is too large for where it is going.

    Raised before the image edit, not after: a replacement that cannot
    plausibly occupy the position produces a result nobody wants, and finding
    that out costs a minute of image generation. The caller decides whether to
    proceed -- the check informs, it does not forbid.
    """

    def __init__(self, reasons: list[str], *, ratio: float | None = None) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons
        self.ratio = ratio


@dataclass(frozen=True)
class Selection:
    detection: Detection
    offers: tuple[Offer, ...]
    # Objects the replacement's region will cover, and which the model is
    # therefore told to preserve. Surfaced so a person can see what is at
    # stake before committing to a swap, rather than discovering afterwards
    # that a table went missing.
    affects: tuple[Detection, ...] = ()


class EditSessionService:
    def __init__(self, db: Session, *, editor, fit: FitEngine | None = None) -> None:
        self.db = db
        self.editor = editor
        self.fit = fit or FitEngine()

    # -------------------------------------------------------------- start

    def start(self, *, scene_id: str, room_id: str, image_ref: str) -> EditSession:
        detections, notes = self.editor.detect(image_ref)
        session = EditSession(
            id=uuid.uuid4().hex,
            scene_id=scene_id,
            room_id=room_id,
            original_image_ref=image_ref,
            detections=[d.to_dict() for d in detections],
        )
        self.db.add(session)
        self.db.flush()
        return session

    def get(self, session_id: str) -> EditSession | None:
        return self.db.get(EditSession, session_id)

    def _detections(self, session: EditSession) -> list[Detection]:
        return [Detection.from_dict(d) for d in session.detections]

    # ------------------------------------------------------------ current

    def current_image(self, session: EditSession) -> str:
        """The image the user is currently looking at."""
        if session.current_step_id is None:
            return session.original_image_ref
        step = self.db.get(EditStep, session.current_step_id)
        return step.result_image_ref if step else session.original_image_ref

    # ------------------------------------------------------------- select

    def select(
        self, session: EditSession, x: int | None = None, y: int | None = None,
        *, detection_id: str | None = None, room: Room | None = None,
        limit: int = 10,
    ) -> Selection | None:
        """Resolve a selection to a detection and gather catalogue offers.

        Two ways in, one result: a click (normalised 0-1000 coordinates,
        resolution-independent) or a direct ``detection_id`` when the user
        picks from the object list instead of the image. Both paths converge
        on the same offers logic.
        """
        if detection_id is not None:
            det = next(
                (d for d in self._detections(session) if d.id == detection_id), None
            )
        elif x is not None and y is not None:
            det = hit_test(self._detections(session), x, y)
        else:
            det = None
        if det is None:
            return None

        stmt = (
            select(CatalogueItemRow)
            .where(
                CatalogueItemRow.object_class == det.object_class,
                CatalogueItemRow.active == 1,
            )
            .limit(limit)
        )
        rows = list(self.db.execute(stmt).scalars())

        offers: list[Offer] = []
        for r in rows:
            fits, note = True, None
            if room is not None and det.object_class in _FIT_CHECKED:
                fits, note = self._fits(r, room)
            tags = r.style_tags if isinstance(r.style_tags, dict) else {}
            offers.append(
                Offer(
                    sku=r.sku,
                    name=r.name,
                    object_class=r.object_class,
                    width_mm=r.width_mm,
                    depth_mm=r.depth_mm,
                    height_mm=r.height_mm,
                    display_price=str(r.display_price),
                    currency=r.currency,
                    image_ref=r.image_ref,
                    fits_room=fits,
                    fit_note=note,
                    suggested=bool(tags.get("suggested")),
                    swatch=tags.get("hex"),
                )
            )
        # Suggested first, then fitting, then the rest with their reasons --
        # non-fitting stock is shown last, never hidden.
        offers.sort(key=lambda o: (not o.fits_room, not o.suggested))

        # Size the region for a representative offer so the warning reflects
        # what a swap would actually touch.
        from .editing import overlapping_detections, replacement_region

        affects: tuple[Detection, ...] = ()
        try:
            probe = offers[0] if offers else None
            dims = (probe.width_mm, probe.depth_mm, probe.height_mm) if probe else None
            floor = next(
                (d for d in self._detections(session) if d.object_class == "floor"),
                None,
            )
            region = replacement_region(
                1000, 1000, det.box, det.object_class, dims,
                label=det.label,
                floor_top=floor.box[1] if floor else None,
            )
            affects = tuple(
                overlapping_detections(
                    region,
                    [d for d in self._detections(session) if d.id != det.id],
                    exclude_id=det.id,
                )
            )
        except Exception:
            affects = ()

        return Selection(detection=det, offers=tuple(offers), affects=affects)

    def _fits(self, row: CatalogueItemRow, room: Room) -> tuple[bool, str | None]:
        """Coarse dimensional feasibility -- can this item fit the room at all,
        at any position/rotation? Cheapest check that answers the picker's
        question; exact placement remains the solver's job."""
        item = CatalogueItem(
            sku=row.sku,
            name=row.name,
            object_class=ObjectClass(row.object_class),
            footprint=Footprint(
                width_mm=row.width_mm, depth_mm=row.depth_mm, height_mm=row.height_mm
            ),
        )
        # Try both orientations at the room's min corner, dimension gates only
        # (skip spatial gates -- position is not decided here).
        minx, miny, _, _ = room.bounds
        for yaw in (0, 90):
            res = self.fit.check(
                item, room, Vec2(x=minx, y=miny), yaw, skip_circulation=True,
                collect_all=True,
            )
            dim_codes = {"TOO_WIDE", "TOO_DEEP", "TOO_TALL"}
            dim_fails = [r for r in res.rejections if r.code.value in dim_codes]
            if not dim_fails:
                return True, None
        return False, dim_fails[0].message if dim_fails else "does not fit"

    # ---------------------------------------------------------- preflight

    #: How much WIDER than the object it replaces a product may be before the
    #: swap is worth questioning.
    #:
    #: Width, not area. A tall bookcase standing in for a squat one needs
    #: several times the area and that is entirely normal -- detection boxes
    #: are tight, and vertical growth is expected and handled. Sideways growth
    #: is the signal that a product does not belong in this spot: it is what
    #: makes a replacement reach across its neighbours.
    OVERSIZE_WIDTH_RATIO = 2.2

    def preflight(
        self, session: EditSession, det: Detection, row: CatalogueItemRow,
        *, room: Room | None = None,
    ) -> list[str]:
        """Reasons this swap looks unwise, in plain words. Empty means fine."""
        from .editing import replacement_region

        reasons: list[str] = []

        # Does it fit the room at all? Same gate the picker uses.
        if room is not None and row.object_class in _FIT_CHECKED:
            fits, note = self._fits(row, room)
            if not fits:
                reasons.append(
                    f"It does not fit the room: {note}"
                    if note else "It is too large for this room."
                )

        # Does it fit the position? Compare the area a correctly-proportioned
        # product would need against the area the current object occupies.
        ratio: float | None = None
        try:
            floor = next(
                (d for d in self._detections(session) if d.object_class == "floor"),
                None,
            )
            region = replacement_region(
                1000, 1000, det.box, det.object_class,
                (row.width_mm, row.depth_mm, row.height_mm),
                label=det.label,
                floor_top=floor.box[1] if floor else None,
            )
            box_w = max(1, det.box[2] - det.box[0])
            region_w = max(1, region[2] - region[0])
            ratio = region_w / box_w
            if ratio >= self.OVERSIZE_WIDTH_RATIO:
                reasons.append(
                    f"It is roughly {ratio:.1f}x wider than the "
                    f"{det.label} it would replace."
                )
        except Exception:
            ratio = None

        if reasons:
            # Name what it would reach across, so the warning is specific
            # rather than just "too big".
            try:
                from .editing import overlapping_detections

                covered = [
                    d.label for d in overlapping_detections(
                        region,
                        [d for d in self._detections(session) if d.id != det.id],
                        exclude_id=det.id,
                    )
                ][:4]
            except Exception:
                covered = []
            if covered:
                reasons.append("It would extend across: " + ", ".join(covered) + ".")

        return reasons

    # -------------------------------------------------------------- apply

    def apply(
        self, session: EditSession, detection_id: str, sku: str,
        *, expand: float = 1.0, redetect: bool = True,
        confirm_oversize: bool = False, room: Room | None = None,
    ) -> EditStep:
        """Replace the selected object with a catalogue item; append a step.

        By default the edited image is re-analysed afterwards so every box and
        label reflects what is actually there now. Pass ``redetect=False`` to
        skip that second model call and rely on the geometric estimate.
        """
        det = next(
            (d for d in self._detections(session) if d.id == detection_id), None
        )
        if det is None:
            raise KeyError(f"no detection {detection_id!r} in session")

        row = self.db.get(CatalogueItemRow, sku)
        if row is None or not row.active:
            raise KeyError(f"no active catalogue item {sku!r}")

        if not confirm_oversize:
            concerns = self.preflight(session, det, row, room=room)
            if concerns:
                raise OversizeReplacement(concerns)

        base_image = self.current_image(session)
        replace_kwargs = {
            "product_name": row.name,
            "product_desc": row.description or "",
            "product_image_ref": row.image_ref,
        }
        # Real dimensions let the editor size the replacement's headroom from
        # the product's own proportions instead of guessing. Passed only when
        # the editor accepts it, so a simpler editor stays compatible.
        import inspect

        params = inspect.signature(self.editor.replace).parameters
        if "product_dims" in params:
            replace_kwargs["product_dims"] = (
                row.width_mm, row.depth_mm, row.height_mm
            )
        if "expand" in params and expand != 1.0:
            replace_kwargs["expand"] = expand
        if "neighbours" in params:
            # Everything else detected in the photo, so the editor can name
            # what must survive the edit rather than trusting the model to
            # infer it.
            replace_kwargs["neighbours"] = [
                d for d in self._detections(session) if d.id != det.id
            ]
        if "floor_top" in params:
            # The floor is itself a detected region, so "how far down does a
            # standing replacement need to reach" has a measured answer rather
            # than a guessed one.
            floor = next(
                (d for d in self._detections(session) if d.object_class == "floor"),
                None,
            )
            if floor is not None:
                replace_kwargs["floor_top"] = floor.box[1]
        result_ref = self.editor.replace(base_image, det, **replace_kwargs)

        # Re-detect on the edited image. The geometric estimate below is
        # derived from the product's dimensions against a class typical, which
        # tracks a like-for-like swap well but not a shape change -- a narrow
        # 2 m bookcase standing in for a wide low one leaves the old outline
        # visibly wrong. Measuring the new image is the honest answer; the
        # estimate stays as the fallback when detection is unavailable.
        previous_detections = list(session.detections)
        previous_geometry: dict = {"previous_detections": previous_detections}
        refreshed = False
        if redetect:
            try:
                from .editing import reconcile_detections

                detected, _notes = self.editor.detect(result_ref)
                if detected:
                    # Identities must survive, or the quote prices the same
                    # object twice: supersession is keyed on detection id.
                    merged = reconcile_detections(self._detections(session), detected)
                    swapped = next((d for d in merged if d.id == det.id), None)
                    # A detector that reports the swapped object completely
                    # unchanged has not seen the edit -- an offline stand-in,
                    # or a pass that missed it. Trusting that would leave the
                    # old name and outline on the new product, so fall through
                    # to the estimate rather than accept an answer that cannot
                    # be right.
                    if swapped is not None and swapped.box == det.box \
                            and swapped.label == det.label:
                        refreshed = False
                    else:
                        session.detections = [d.to_dict() for d in merged]
                        flag_modified(session, "detections")
                        refreshed = True
            except Exception:
                refreshed = False

        if not refreshed:
            try:
                from .editing import replaced_object_box

                floor = next(
                    (d for d in self._detections(session) if d.object_class == "floor"),
                    None,
                )
                new_box = replaced_object_box(
                    det.box, det.object_class,
                    (row.width_mm, row.depth_mm, row.height_mm),
                    label=det.label,
                    floor_top=floor.box[1] if floor else None,
                )
                if new_box != det.box:
                    updated = []
                    for entry in session.detections:
                        if entry.get("id") == det.id:
                            entry = {**entry, "box": list(new_box), "label": row.name}
                        updated.append(entry)
                    session.detections = updated
                    flag_modified(session, "detections")
            except Exception:
                # Geometry bookkeeping must never lose a completed replacement.
                pass

        step = EditStep(
            notes=previous_geometry,
            session_id=session.id,
            parent_step_id=session.current_step_id,
            detection_id=det.id,
            detection_label=det.label,
            replacement_sku=sku,
            result_image_ref=result_ref,
            provider=type(self.editor).__name__,
        )
        self.db.add(step)
        self.db.flush()
        session.current_step_id = step.id
        self.db.flush()
        return step

    def redetect(self, session: EditSession) -> tuple[int, list[str]]:
        """Re-run detection on the CURRENT image.

        After a swap the stored boxes are estimated from the product's
        proportions -- close, but a guess. Re-detecting measures what is
        actually in the picture now, which matters most when a replacement
        changed an object's shape a lot, or when the model moved things
        slightly while rendering.

        Costs one detection call. The step chain is untouched, so previous
        swaps and the quote built from them survive: those are keyed to the
        steps, not to the detections being replaced here.
        """
        detections, notes = self.editor.detect(self.current_image(session))
        if not detections:
            return 0, notes or ["detection returned nothing; keeping the previous boxes"]
        session.detections = [d.to_dict() for d in detections]
        flag_modified(session, "detections")
        self.db.flush()
        return len(detections), notes

    # ------------------------------------------------------- instructions

    def interpret(
        self, session: EditSession, text: str,
        *, detection_id: str | None = None,
    ):
        """Understand a typed request without acting on it."""
        selected = None
        if detection_id:
            selected = next(
                (d for d in self._detections(session) if d.id == detection_id),
                None,
            )
        return self.editor.analyse_instruction(
            text, self._detections(session), selected=selected
        )

    def instruct(
        self, session: EditSession, text: str,
        *, detection_id: str | None = None, confirm_mismatch: bool = False,
    ) -> "tuple[EditStep | None, object]":
        """Carry out a typed request. Returns (step, intent).

        ``step`` is None when nothing was done -- either the request could not
        be understood, or the words describe a different object than the one
        clicked and that needs confirming. A misplaced click is common, and
        silently editing whatever was selected would be worse than asking.
        """
        from .editing import EditIntent, reconcile_detections

        if not detection_id:
            # Nothing selected: hand the request straight to the image model
            # with the photograph and no object map at all.
            #
            # The map actively harms this case. Detection carves a room's
            # walls into separate regions -- "marble feature wall" behind the
            # television, painted walls either side -- and any target resolved
            # from that list inherits the carving, so "paint the wall" repaints
            # one panel. Those boundaries are an artefact of how detection
            # segments a photo, not of how a person sees the room. Without the
            # map the model reads the request against the whole image, which
            # is what someone describing a change to their room means.
            intent = EditIntent(
                target_ids=(),
                operation="scene",
                instruction=text.strip(),
                confidence=1.0,
                selection_matches=None,
                note="",
            )
        else:
            intent = self.interpret(session, text, detection_id=detection_id)
            if not intent.is_actionable:
                return None, intent
            if intent.selection_matches is False and not confirm_mismatch:
                return None, intent

        detections = self._detections(session)
        by_id = {d.id: d for d in detections}
        group = [by_id[i] for i in intent.target_ids if i in by_id]
        target = group[0] if group else None

        floor = next((d for d in detections if d.object_class == "floor"), None)
        base_image = self.current_image(session)
        group_ids = {d.id for d in group}
        replace_kwargs = {
            "target": target,
            # Only meaningful alongside a target: they tell a region-locked
            # edit what it must not damage. With no target there is no region,
            # so passing them would just be describing the room to a model
            # that can already see it.
            "neighbours": (
                [d for d in detections if d.id not in group_ids] if group else []
            ),
            "floor_top": floor.box[1] if floor else None,
        }
        import inspect

        if "targets" in inspect.signature(self.editor.instruct).parameters:
            replace_kwargs["targets"] = group
        result_ref = self.editor.instruct(base_image, intent, **replace_kwargs)

        previous_geometry = {"previous_detections": list(session.detections)}
        if target is not None:
            try:
                detected, _notes = self.editor.detect(result_ref)
                if detected:
                    merged = reconcile_detections(detections, detected)
                    session.detections = [d.to_dict() for d in merged]
                    flag_modified(session, "detections")
            except Exception:
                pass

        step = EditStep(
            session_id=session.id,
            parent_step_id=session.current_step_id,
            detection_id=target.id if target is not None else "scene",
            detection_label=target.label if target is not None else "whole scene",
            replacement_sku=None,
            instruction=text.strip(),
            result_image_ref=result_ref,
            provider=type(self.editor).__name__,
            notes=previous_geometry,
        )
        self.db.add(step)
        self.db.flush()
        session.current_step_id = step.id
        self.db.flush()
        return step, intent

    # ---------------------------------------------------------- undo/redo

    def undo(self, session: EditSession) -> str:
        """Move the pointer one step back. Returns the now-current image.

        Also restores the outline the undone swap changed: reverting the image
        while leaving the new product's box in place would draw a boundary
        around something no longer in the picture.
        """
        if session.current_step_id is None:
            return session.original_image_ref
        step = self.db.get(EditStep, session.current_step_id)
        if step is not None and isinstance(step.notes, dict):
            # The whole detection set is restored, not just the swapped
            # object's box: a re-detection may have adjusted every outline, so
            # reverting one of them would leave the rest describing an image
            # that no longer exists.
            snapshot = step.notes.get("previous_detections")
            if snapshot:
                session.detections = list(snapshot)
                flag_modified(session, "detections")
        session.current_step_id = step.parent_step_id if step else None
        self.db.flush()
        return self.current_image(session)

    # ----------------------------------------------------------- manifest

    def change_manifest(self, session: EditSession) -> dict:
        """Everything that was changed, and what is known about the cost.

        Two kinds of change end up in a quote and they must not be confused:

        * **Known** -- a catalogue product was swapped in. We have the SKU, the
          price and the vendor. These are facts, not estimates, and the model
          is told to use them verbatim rather than price a sofa it can only
          guess at.
        * **Estimated** -- a typed instruction ("paint the walls sage"), where
          nothing is known but the words. Materials, labour and rates all have
          to be estimated for the region.

        Keeping the two apart is the same discipline as ``estimated_prior`` on
        room dimensions: a guess is never allowed to look like a measurement.
        """
        chain: list[EditStep] = []
        step_id = session.current_step_id
        while step_id is not None:
            step = self.db.get(EditStep, step_id)
            if step is None:
                break
            chain.append(step)
            step_id = step.parent_step_id

        known: list[dict] = []
        instructions: list[dict] = []
        for step in reversed(chain):
            if step.replacement_sku:
                row = self.db.get(CatalogueItemRow, step.replacement_sku)
                if row is None:
                    continue
                known.append({
                    "sku": row.sku,
                    "name": row.name,
                    "object_class": row.object_class,
                    "description": row.description or "",
                    "replaced": step.detection_label,
                    "width_mm": row.width_mm,
                    "depth_mm": row.depth_mm,
                    "height_mm": row.height_mm,
                    "price": str(row.display_price),
                    "currency": row.currency,
                    "vendor": row.vendor or "catalogue",
                })
            elif step.instruction:
                instructions.append({
                    "instruction": step.instruction,
                    "applied_to": step.detection_label,
                })

        # A later swap of the same object supersedes an earlier one, so only
        # what is in the final image should be priced.
        current = set(self.swapped_skus(session).values())
        known = [k for k in known if k["sku"] in current]

        return {"known_products": known, "instructions": instructions}

    # -------------------------------------------------------------- quote

    def swapped_skus(self, session: EditSession) -> dict[str, str]:
        """What the CURRENT image contains: {detection_id: sku}.

        Walks the chain from the current step back to the original. A later
        swap of the same detection supersedes an earlier one -- if the user
        tried three sofas, only the one in the final image is priced.
        """
        result: dict[str, str] = {}
        chain: list[EditStep] = []
        step_id = session.current_step_id
        while step_id is not None:
            step = self.db.get(EditStep, step_id)
            if step is None:
                break
            chain.append(step)
            step_id = step.parent_step_id
        # chain is newest-first; walk oldest-first so later swaps overwrite.
        for step in reversed(chain):
            if step.replacement_sku is None:
                # A free-text edit changed the photo but added no product, so
                # it must not appear on the quote. If it changed an object
                # that HAD been swapped, that product is no longer what is in
                # the picture either.
                result.pop(step.detection_id, None)
                continue
            result[step.detection_id] = step.replacement_sku
        return result