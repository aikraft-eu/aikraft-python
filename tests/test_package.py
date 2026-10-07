import aikraft


def test_version_is_exposed() -> None:
    assert isinstance(aikraft.__version__, str) and aikraft.__version__
