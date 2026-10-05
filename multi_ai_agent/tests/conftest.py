from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.core.container import Container
from tests.fakes import FakeFactory, build_container, make_settings


@pytest.fixture
def factory() -> FakeFactory:
    return FakeFactory()


@pytest.fixture
async def container(tmp_path, factory) -> AsyncIterator[Container]:
    c = build_container(make_settings(tmp_path), factory=factory)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()
