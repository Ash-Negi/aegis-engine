"""
Aegis Engine — pytest configuration
===================================

`conftest.py` is a special file that pytest auto-discovers during test
collection. You never import it — pytest finds it by walking up from each
test file. This one file does several jobs:

1. Path injection (the reason this file exists right now).
   pytest prepends any directory containing a conftest.py to sys.path.
   Placing this file at the `math-engine/` level lets tests under
   `tests/` resolve imports like `from config import ...` and
   `from data.pipeline import ...` without needing to cd into math-engine
   or install the project as a package.

2. Shared fixtures.
   Fixtures defined here are available to every test file in this
   directory and below — no import needed. If the same fixture (e.g. a
   cached Dataset, a tmp data dir) starts getting copy-pasted across
   test modules, move it here.

3. Hooks and plugins.
   pytest hooks like `pytest_collection_modifyitems`, `pytest_configure`,
   or custom CLI options live here. Not needed yet.

4. Scope by location.
   A conftest.py in `tests/` would apply only to tests under `tests/`.
   This one, at the project root of `math-engine/`, applies project-wide.
   Nested conftests compose — inner ones extend outer ones.

Leaving this file intentionally empty below. Its mere presence does the
sys.path work for us.
"""
