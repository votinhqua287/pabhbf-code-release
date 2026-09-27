"""Tests that quote the manuscript are skipped when the LaTeX sources are not present."""
import inspect

import pytest

from pabhbf.utils.config import ROOT

MANUSCRIPT = ROOT / "manuscript" / "main.tex"


def pytest_collection_modifyitems(config, items):
    if MANUSCRIPT.exists():
        return
    skip = pytest.mark.skip(reason="manuscript sources are not part of the code package")
    for item in items:
        function = getattr(item, "function", None)
        if function is None:
            continue
        try:
            source = inspect.getsource(function)
        except OSError:
            continue
        if "main.tex" in source or "results_text(" in source:
            item.add_marker(skip)
