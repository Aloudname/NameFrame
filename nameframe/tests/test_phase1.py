"""end-to-end test exercising the full phase 1 public api."""

from nameframe import __version__
from nameframe.registry import (
    Registry,
    RegistryKeyError,
    dataset,
    discover,
    loss,
    metric,
    model,
    ops,
    plugin,
)
from nameframe.utils import (
    BatchProtocol,
    FieldSchema,
    capture_env,
    derive_seed,
    set_seed,
)


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__.count(".") == 2


def test_all_registry_instances_exist():
    for reg in [model, dataset, loss, metric, ops, plugin]:
        assert isinstance(reg, Registry)
        assert len(reg) >= 0


def test_full_registration_flow():
    """decorator -> get -> contains -> list_all"""
    @model.register("integration_test_model")
    class TestModel:
        pass

    assert "integration_test_model" in model
    assert model.get("integration_test_model") is TestModel
    assert "integration_test_model" in model.list_all()


def test_namespace_isolation():
    ns_a = model.create_namespace("ns_a")
    ns_b = model.create_namespace("ns_b")

    @ns_a.register("same_name")
    class ModelA:
        pass

    @ns_b.register("same_name")
    class ModelB:
        pass

    assert model.get("ns_a:same_name") is ModelA
    assert model.get("ns_b:same_name") is ModelB
    assert model.get("ns_a:same_name") is not model.get("ns_b:same_name")


def test_registry_key_error_is_helpful():
    try:
        model.get("definitely_not_registered_xyz")
        assert False, "should have raised"
    except RegistryKeyError as e:
        assert "definitely_not_registered_xyz" in str(e)
        assert "model" in str(e)


def test_seed_and_env_together():
    set_seed(123)
    env = capture_env()
    assert "torch" in env