import pytest

import gorilla


def test_registry():
    OBJECT_REGISTRY = gorilla.Registry("OBJECT")

    @OBJECT_REGISTRY.register()
    class Object1:
        pass

    with pytest.raises(KeyError):
        OBJECT_REGISTRY.register(Object1)

    assert OBJECT_REGISTRY.get("Object1") == Object1

    with pytest.raises(KeyError):
        OBJECT_REGISTRY.get("Object2")

    assert list(OBJECT_REGISTRY) == [("Object1", Object1)]


