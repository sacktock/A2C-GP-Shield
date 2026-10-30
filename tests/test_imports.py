"""Import surface for the pinned scientific stack.

These stay independent of gpshield internals so a GPJAX API migration
cannot hide a broken dependency install.
"""

import importlib

import pytest

REQUIRED_MODULES = [
    "numpy",
    "gpjax",
    "jax",
    "jaxlib",
    "tensorflow",
]


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_module_imports(module_name):
    importlib.import_module(module_name)
