"""
Benchmarking Tools - Nástroje pro porovnání úspory před/po optimalizaci
"""
import time
from typing import Callable, Dict, Any
import logging

class BenchmarkingTools:
    """Nástroje pro A/B testování a benchmarking optimalizačních strategií"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Benchmarkování výkonu funkce"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        self.logger.info(f"Funkce {func.__name__} vykonána za {execution_time:.4f} sekund.")

        return {
            "result": result,
            "execution_time": execution_time
        }

    def ab_test(self, func_a: Callable, func_b: Callable, *args, **kwargs) -> Dict[str, Any]:
        """A/B testování dvou funkcí"""
        self.logger.info("Spouštím A/B testování...")

        result_a = self.benchmark_function(func_a, *args, **kwargs)
        result_b = self.benchmark_function(func_b, *args, **kwargs)

        comparison = {
            "func_a_time": result_a["execution_time"],
            "func_b_time": result_b["execution_time"],
            "winner": "func_a" if result_a["execution_time"] < result_b["execution_time"] else "func_b"
        }

        self.logger.info(f"Výsledek A/B testu: {comparison}")
        return comparison
