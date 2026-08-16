"""spark-submit entrypoint shim.

spark-submit runs its target .py file as a bare script, which gives it no
parent package -- so the relative imports used throughout ``tickets.spark.*``
(``from ..config import ...``) fail with "attempted relative import with no
known parent package". Running the target as an actual module import instead
of executing the file directly gives Python the real package context, the
same way ``python -m tickets.spark.stream_job`` would if spark-submit
supported ``-m``.

Usage: spark-submit run_module.py <module.dotted.path> [args...]
"""

from __future__ import annotations

import runpy
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_module.py <module.dotted.path> [args...]")
    module = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_module(module, run_name="__main__")
