def test_package_functionality():
    """Test basic functionality of the installed package."""
    import buildstock_fetch

    assert hasattr(buildstock_fetch, "__version__"), "Package should have __version__"
