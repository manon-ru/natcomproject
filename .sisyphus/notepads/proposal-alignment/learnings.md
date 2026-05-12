2026-05-12: time_to_half_entropy must measure from peak_index and scale by sample_interval.
[2026-05-12] adaptation_time floor behavior
- Used an explicit entropy_floor parameter rather than config import to keep the metric self-contained.
- Kept threshold_ratio=0.8 and sample_interval=10 unchanged.
4. [2026-05-12] ES baseline removal
- Removed OnePlusOneES from the package exports and main runner.
- Verified with a direct __all__ check and grep-based reference count.
