from asme.core.init.factories.include.import_factory import ImportFactory


# This makes sure that all default processors, datasets, modules, etc. are imported before tests are run
def pytest_sessionstart(session):
    import importlib
    for module in ImportFactory.DEFAULT_MODULES:
        importlib.import_module(module, ImportFactory.TOP_LEVEL_MODULE_NAME)
