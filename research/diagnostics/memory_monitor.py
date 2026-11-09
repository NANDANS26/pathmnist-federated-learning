"""
Hardware Memory and System Resource Monitoring.
Protects dual-core 4 GB RAM Windows host from OS thrashing and pagefile collapse.
"""

import os
import psutil
import gc
from typing import Dict, Tuple


class MemoryPressureException(Exception):
    """Raised when available system physical memory drops below critical safety threshold."""
    pass


class MemoryMonitor:
    """
    Monitors process RSS and system available physical RAM.
    Triggers emergency exceptions if physical memory approaches system lockup.
    """
    def __init__(self, emergency_ram_floor_mb: float = 150.0, reference_rss_mb: float = 1200.0):
        self.emergency_ram_floor_mb = emergency_ram_floor_mb
        self.reference_rss_mb = reference_rss_mb
        self.process = psutil.Process(os.getpid())
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Returns current process RSS and system memory metrics in megabytes (MB)."""
        vm = psutil.virtual_memory()
        proc_info = self.process.memory_info()
        
        rss_mb = proc_info.rss / (1024.0 * 1024.0)
        avail_mb = vm.available / (1024.0 * 1024.0)
        total_mb = vm.total / (1024.0 * 1024.0)
        percent_used = vm.percent
        
        return {
            "process_rss_mb": float(round(rss_mb, 2)),
            "system_available_mb": float(round(avail_mb, 2)),
            "system_total_mb": float(round(total_mb, 2)),
            "system_percent_used": float(round(percent_used, 1))
        }
        
    def check_safety(self) -> Dict[str, float]:
        """
        Queries memory state and asserts that available physical RAM remains above emergency threshold.
        Raises MemoryPressureException if system RAM drops below floor.
        """
        stats = self.get_memory_stats()
        
        if stats["system_available_mb"] < self.emergency_ram_floor_mb:
            # Force emergency GC cleanup to attempt recovery
            gc.collect()
            recovered_stats = self.get_memory_stats()
            if recovered_stats["system_available_mb"] < self.emergency_ram_floor_mb:
                raise MemoryPressureException(
                    f"CRITICAL MEMORY PRESSURE: Available physical RAM ({recovered_stats['system_available_mb']:.1f} MB) "
                    f"dropped below emergency safety floor ({self.emergency_ram_floor_mb:.1f} MB)! "
                    f"Process RSS: {recovered_stats['process_rss_mb']:.1f} MB. Halting immediately."
                )
        return stats
