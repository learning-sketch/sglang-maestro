"""Performance analysis: roofline estimates, profiler-trace analysis and server-log parsing."""

from sgl_agent.perf.analyze import TraceReport, analyze_parsed_trace, analyze_trace, load_dialect
from sgl_agent.perf.logs import LogReport, analyze_logs, parse_logs
from sgl_agent.perf.roofline import RooflineReport, analyze_roofline, default_scenarios

__all__ = [
    "LogReport",
    "RooflineReport",
    "TraceReport",
    "analyze_logs",
    "analyze_parsed_trace",
    "analyze_roofline",
    "analyze_trace",
    "default_scenarios",
    "load_dialect",
    "parse_logs",
]
