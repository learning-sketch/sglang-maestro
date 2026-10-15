"""Synthetic SGLang server logs and benchmark records for demos and tests.

Step times are the cost model's times multiplied by a known slowdown (``decode_scale``,
``prefill_scale``), so ``sgl-agent calibrate`` on the output should recover those factors. Lines
follow the formats ``perf.logs`` parses; the first line marks the file as synthetic and the
benchmark record carries ``"synthetic": true``. These are not measurements.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge.facts import DEFAULT_MAX_PREFILL_TOKENS
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import BatchShape, Calibration, ExecutionConfig, StepCostModel
from sgl_agent.model.parallel import ParallelConfig, plan_memory
from sgl_agent.perf.logs import SYNTHETIC_LOG_MARK

DECODE_LOG_INTERVAL = 40
START = datetime(2026, 1, 1, 12, 0, 0)


class _Writer:
    def __init__(self) -> None:
        self.clock = START
        self.lines = [SYNTHETIC_LOG_MARK]

    def emit(self, message: str) -> None:
        stamp = f"{self.clock:%Y-%m-%d %H:%M:%S}.{self.clock.microsecond // 1000:03d}"
        self.lines.append(f"[{stamp}] {message}")

    def advance(self, microseconds: float) -> None:
        self.clock += timedelta(microseconds=microseconds)


def synthetic_server_log(
    arch: ModelArch,
    hardware: HardwareSpec,
    parallel: ParallelConfig,
    *,
    decode_scale: float = 1.3,
    prefill_scale: float = 1.15,
    bursts: int = 3,
    burst_batches: int = 6,
    decode_lines: int = 10,
    max_batch: int = 96,
    prompt_len: int = 2048,
    output_len: int = 512,
    execution: ExecutionConfig | None = None,
    seed: int = 1,
) -> list[str]:
    """A startup line, then ``bursts`` rounds of back-to-back prefill batches and decode logs."""
    if min(decode_scale, prefill_scale) <= 0:
        raise ValueError("scales must be > 0")
    cost = StepCostModel(arch, hardware, parallel, execution or ExecutionConfig(), Calibration())
    memory = plan_memory(arch, hardware, parallel)
    capacity = max(memory.max_total_tokens, 1)
    chunk = memory.chunked_prefill_size
    rng = random.Random(seed)
    out = _Writer()
    out.emit(
        f"max_total_num_tokens={capacity}, chunked_prefill_size={chunk}, "
        f"max_prefill_tokens={DEFAULT_MAX_PREFILL_TOKENS}, max_running_requests={max_batch}, "
        f"context_len={arch.max_position_embeddings or 4096}, available_gpu_mem=8.00 GB"
    )
    running = 0
    used = 0
    for _ in range(bursts):
        queue = burst_batches * 4
        previous_us = None
        for _ in range(burst_batches):
            seqs = rng.choice((1, 2, 4))
            per_seq = max(min(prompt_len, chunk // seqs if chunk > 0 else prompt_len), 1)
            cached = rng.choice((0, 0, per_seq // 2))
            tokens = per_seq * seqs
            step_us = (
                cost.estimate(
                    BatchShape.chunks([(per_seq, cached)] * seqs), cuda_graph=False
                ).total_us
                * prefill_scale
            )
            gap = previous_us if previous_us is not None else step_us * 3
            queue = max(queue - seqs, 1)
            running += seqs
            used += (per_seq + cached) * seqs
            out.emit(
                f"Prefill batch, #new-seq: {seqs}, #new-token: {tokens}, #cached-token: "
                f"{cached * seqs}, token usage: {min(used / capacity, 1.0):.2f}, #running-req: "
                f"{running - seqs}, #queue-req: {queue}, #pending-token: 0, cuda graph: False, "
                f"input throughput (token/s): {tokens / gap * 1e6:.2f}"
            )
            out.advance(step_us)
            previous_us = step_us
        running = min(running, max_batch)
        for line in range(decode_lines + 1):
            context = prompt_len + output_len * (line + 1) / (decode_lines + 2)
            kv = int(min(running * context, capacity))
            step_us = (
                cost.estimate(BatchShape.decode_total(running, kv), cuda_graph=True).total_us
                * decode_scale
            )
            out.advance(step_us * DECODE_LOG_INTERVAL)
            out.emit(
                f"Decode batch, #running-req: {running}, #token: {kv}, token usage: "
                f"{kv / capacity:.2f}, cuda graph: True, gen throughput (token/s): "
                f"{running / step_us * 1e6:.2f}, #queue-req: 0"
            )
        running = rng.randint(max_batch // 4, max_batch // 2)
        used = running * prompt_len
    return out.lines


def synthetic_bench_record(
    result: dict[str, Any], *, parallel: ParallelConfig, request_rate: float | None
) -> dict[str, Any]:
    """A ``bench_serving --output-file`` record from a simulation result (``SimResult.to_dict``)."""
    ttft, tpot = result["ttft_ms"], result["tpot_ms"]
    duration = result["duration_s"]
    return {
        "synthetic": True,
        "backend": "sglang",
        "request_rate": request_rate if request_rate is not None else "inf",
        "server_info": {
            "tp_size": parallel.tp,
            "pp_size": parallel.pp,
            "attn_dp_size": parallel.attn_dp,
            "attn_cp_size": parallel.attn_cp,
            "dcp_size": parallel.dcp,
            "ep_size": parallel.ep,
        },
        "duration": duration,
        "completed": result["completed"],
        "request_throughput": result["request_throughput"],
        "output_throughput": result["output_throughput"],
        "mean_ttft_ms": result["mean_ttft_ms"],
        "median_ttft_ms": ttft["p50"],
        "p90_ttft_ms": ttft["p90"],
        "p99_ttft_ms": ttft["p99"],
        "mean_tpot_ms": result["mean_tpot_ms"],
        "median_tpot_ms": tpot["p50"],
        "p90_tpot_ms": tpot["p90"],
        "p99_tpot_ms": tpot["p99"],
        "mean_e2e_latency_ms": result["mean_e2e_ms"],
        "concurrency": result["mean_e2e_ms"] * result["completed"] / 1000 / duration
        if duration
        else None,
    }


def write_synthetic_log(lines: list[str], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_bench_record(record: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return target
