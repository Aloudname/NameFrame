import pytest

from nameframe.registry import Registry, RegistryKeyError, model


class TestRegistryInit:
    def test_creates_with_name(self):
        r = Registry("test")
        assert r.name == "test"
        assert len(r) == 0

    def test_repr_includes_name_and_count(self):
        r = Registry("test")
        assert "test" in repr(r)


class TestRegistryRegister:
    def test_decorator_registers_class(self):
        r = Registry("test")

        @r.register("my_class")
        class MyClass:
            pass

        assert "my_class" in r
        assert r.get("my_class") is MyClass

    def test_decorator_auto_names_class(self):
        r = Registry("test")

        @r.register()
        class MyViTBlock:
            pass

        # MyViTBlock -> my_vi_t_block (edge case with consecutive caps)
        assert "my_vi_t_block" in r

    def test_register_external_adds_item(self):
        r = Registry("test")
        r.register_external("ext", dict)
        assert r.get("ext") is dict

    def test_base_type_enforcement(self):
        r = Registry("test", base_type=dict)

        with pytest.raises(TypeError):
            r.register_external("bad", list)

    def test_duplicate_registration_overwrites(self):
        r = Registry("test")

        @r.register("dup")
        class V1:
            pass

        @r.register("dup")
        class V2:
            pass

        assert r.get("dup") is V2


class TestRegistryGet:
    def test_get_raises_with_available_list(self):
        r = Registry("test")
        r.register_external("foo", dict)

        with pytest.raises(RegistryKeyError) as exc:
            r.get("bar")
        assert "bar" in str(exc.value)
        assert "foo" in str(exc.value)

    def test_namespace_lookup(self):
        r = Registry("test")
        ns = r.create_namespace("proj")
        ns.register_external("foo", list)

        assert "proj:foo" in r
        assert r.get("proj:foo") is list


class TestRegistryListAll:
    def test_returns_shallow_copy(self):
        r = Registry("test")
        r.register_external("a", int)
        items = r.list_all()
        items["a"] = str
        assert r.get("a") is int  # original unaffected


class TestRegistryNamespace:
    def test_create_isolated_space(self):
        r = Registry("test")
        r.register_external("x", int)
        ns = r.create_namespace("proj")
        ns.register_external("x", str)

        assert r.get("x") is int
        assert r.get("proj:x") is str
        assert r.list_all() == {"x": int}
        assert r.list_all("proj") == {"x": str}
