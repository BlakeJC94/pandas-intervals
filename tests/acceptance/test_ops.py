from tests.acceptance.conftest import skipif_pytest_acceptance_not_set


@skipif_pytest_acceptance_not_set
class TestFoo:
    def test_foo(self):
        assert False
