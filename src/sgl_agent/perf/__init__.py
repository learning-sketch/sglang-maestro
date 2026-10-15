"""Performance analysis: roofline estimates, profiler-trace analysis, server-log parsing and
cost-model calibration."""

from sgl_agent.perf.analyze import TraceReport, analyze_parsed_trace, analyze_trace, load_dialect
from sgl_agent.perf.calibrate import (
    CalibrationReport,
    calibration_from_dict,
    calibration_to_dict,
    fit_calibration,
    load_bench_records,
    load_calibration,
    render_calibration,
)
from sgl_agent.perf.logs import LogReport, analyze_logs, parse_logs
from sgl_agent.perf.roofline import RooflineReport, analyze_roofline, default_scenarios
from sgl_agent.perf.synthetic_log import (
    synthetic_bench_record,
    synthetic_server_log,
    write_bench_record,
    write_synthetic_log,
)

__all__ = [
    "CalibrationReport",
    "LogReport",
    "RooflineReport",
    "TraceReport",
    "analyze_logs",
    "analyze_parsed_trace",
    "analyze_roofline",
    "analyze_trace",
    "calibration_from_dict",
    "calibration_to_dict",
    "default_scenarios",
    "fit_calibration",
    "load_bench_records",
    "load_calibration",
    "load_dialect",
    "parse_logs",
    "render_calibration",
    "synthetic_bench_record",
    "synthetic_server_log",
    "write_bench_record",
    "write_synthetic_log",
]
