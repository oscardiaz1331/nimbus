"""Assembles every router under the /api/v1 prefix."""

from fastapi import APIRouter

from . import dashboard, health, nodes, observations, providers, stations

router = APIRouter(prefix="/api/v1")
router.include_router(observations.router)
router.include_router(stations.router)
router.include_router(nodes.router)
router.include_router(providers.router)
router.include_router(dashboard.router)
router.include_router(health.router)
