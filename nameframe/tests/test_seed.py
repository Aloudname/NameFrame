from nameframe.utils.seed import derive_seed, set_seed


class TestSetSeed:
    def test_does_not_raise(self):
        set_seed(42)

    def test_is_deterministic(self):
        import random
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b


class TestDeriveSeed:
    def test_different_components_get_different_seeds(self):
        s1 = derive_seed(42, "model_init")
        s2 = derive_seed(42, "data_shuffle")
        assert s1 != s2

    def test_same_inputs_same_output(self):
        a = derive_seed(42, "dropout", rank=0)
        b = derive_seed(42, "dropout", rank=0)
        assert a == b

    def test_different_ranks_different_seeds(self):
        s0 = derive_seed(42, "init", rank=0)
        s1 = derive_seed(42, "init", rank=1)
        assert s0 != s1