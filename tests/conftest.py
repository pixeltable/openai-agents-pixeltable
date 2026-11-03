"""Test isolation: tests run in their own Pixeltable catalog, never ~/.pixeltable.

Set before any test module imports pixeltable. The home is stable per pytest-xdist
worker so later runs reuse the embedded Postgres server.
"""

import os
import tempfile

_worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
os.environ["PIXELTABLE_HOME"] = os.path.join(tempfile.gettempdir(), f"openai-agents-pixeltable-tests-{_worker}")
_runtime = os.path.join(tempfile.gettempdir(), "openai-agents-pixeltable-runtime")
os.makedirs(_runtime, exist_ok=True)
os.chmod(_runtime, 0o700)
os.environ.setdefault("XDG_RUNTIME_DIR", _runtime)
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
os.environ.pop("OPENAI_API_KEY", None)
