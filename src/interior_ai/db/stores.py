"""Session-scoped persistence adapters.

The repositories in :mod:`interior_ai.db.repository` each take a live
``Session``. That suits a script or a test, but a web process must not share
one session across requests -- a single dropped connection would poison every
later call, and Neon drops idle connections by design.

These adapters take the *sessionmaker* instead and open a short session per
operation, while exposing exactly the interfaces the API and orchestrator
already use (:class:`SceneStore` and :class:`PriceBook`). That keeps the swap
between in-memory and persistent storage invisible to everything upstream:
the endpoints do not know or care which they are talking to.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session, sessionmaker

from ..core.scene import Scene
from ..pricing.prices import PriceObservation, PriceSnapshot
from .repository import SceneRepository, SqlPriceBook


class SqlSceneStore:
    """Persistent scene storage with the in-memory ``SceneStore`` interface.

    Versions are appended, never replaced -- the same immutability the scene
    graph promises, now surviving a restart.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def put(self, scene: Scene) -> Scene:
        with self._sf() as db:
            SceneRepository(db).save(scene)
            db.commit()
        return scene

    def get(self, scene_id: str) -> Scene:
        with self._sf() as db:
            scene = SceneRepository(db).get_latest(scene_id)
        if scene is None:
            raise KeyError(scene_id)
        return scene

    def get_version(self, version_id: str) -> Scene:
        with self._sf() as db:
            scene = SceneRepository(db).get_version(version_id)
        if scene is None:
            raise KeyError(version_id)
        return scene

    def all_versions(self, scene_id: str) -> list[Scene]:
        with self._sf() as db:
            repo = SceneRepository(db)
            rows = repo.list_versions(scene_id)
            scenes = [repo.get_version(r.version_id) for r in rows]
        return [s for s in scenes if s is not None]


class SqlPriceBookAdapter:
    """Persistent price book with the in-memory ``PriceBook`` interface.

    Prices are the one thing a quote cannot be honest without, so they belong
    in the database rather than a process that forgets them on restart.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def record(self, obs: PriceObservation) -> None:
        with self._sf() as db:
            SqlPriceBook(db).record(obs)
            db.commit()

    def record_many(self, observations: list[PriceObservation]) -> None:
        with self._sf() as db:
            SqlPriceBook(db).record_many(observations)
            db.commit()

    def history_for(self, sku: str) -> tuple[PriceObservation, ...]:
        with self._sf() as db:
            return SqlPriceBook(db).history_for(sku)

    def current(self, sku: str) -> PriceObservation | None:
        with self._sf() as db:
            return SqlPriceBook(db).current(sku)

    def snapshot(self, sku: str, *, as_of: datetime | None = None) -> PriceSnapshot:
        with self._sf() as db:
            return SqlPriceBook(db).snapshot(sku, as_of=as_of)

    def snapshot_many(
        self, skus, *, as_of: datetime | None = None
    ) -> dict[str, PriceSnapshot]:
        with self._sf() as db:
            return SqlPriceBook(db).snapshot_many(skus, as_of=as_of)