# python/research/benchmark_ext.py
"""
Wrapper that delegates to the legacy benchmark_etf_comparison.py script.
Keeps the old CLI entrypoint accessible from python/research.
"""
import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    script = root / "benchmark_etf_comparison.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
