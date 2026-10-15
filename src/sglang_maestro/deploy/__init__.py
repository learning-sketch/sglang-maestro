"""Scenario 5: deploy a tuned model for a workload (instance sizing, replicas, PD, router)."""

from sglang_maestro.deploy.instance import InstanceModel, InstanceSpec, SpecConfig, build_instance
from sglang_maestro.deploy.plan import (
    Candidate,
    Capacity,
    DeployOptions,
    DeployPlan,
    Placement,
    Pool,
    place,
    plan_deployment,
)
from sglang_maestro.deploy.report import render_deploy_plan
from sglang_maestro.deploy.simulate import ClusterSim, SimResult

__all__ = [
    "Candidate",
    "Capacity",
    "ClusterSim",
    "DeployOptions",
    "DeployPlan",
    "InstanceModel",
    "InstanceSpec",
    "Placement",
    "Pool",
    "SimResult",
    "SpecConfig",
    "build_instance",
    "place",
    "plan_deployment",
    "render_deploy_plan",
]
