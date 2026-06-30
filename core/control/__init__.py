"""
core/control/__init__.py
========================
Sorter Control Module - Public API

Hardware Architecture (from ASABE AIM26 poster):
─────────────────────────────────────────────────
3-lane screw conveyor → 3 sorting units (one per lane)

Each sorting unit:
  ┌─────────────────────────────────────────────────┐
  │  NIR switch ──→ detects apple arrival at gate   │
  │  Arduino board ──→ receives PC command via USB  │
  │  Submodule A:                                    │
  │    Solenoid valve A → Air cylinder A → Paddle A │
  │    Deflects apple to Outlet A (FRESH)            │
  │  Submodule B:                                    │
  │    Solenoid valve B → Air cylinder B → Paddle B │
  │    Deflects apple to Outlet B (PROCESSING)      │
  │  Default (no signal):                            │
  │    Apple passes through → Outlet C (CULL)        │
  └─────────────────────────────────────────────────┘

Communication Path:
  PC (Python/PySerial) → USB/Serial → Arduino → Solenoid valves

Timing Model:
  Apple graded at time T (camera)
  Apple arrives at gate at T + delay_ms
  delay_ms = (camera_to_gate_m / conveyor_speed_m_s) * 1000
  Software fires command at T + delay_ms - valve_lead_ms
"""

from core.control.sorter_controller import SorterController, GradeCommand, SortOutlet

__all__ = ["SorterController", "GradeCommand", "SortOutlet"]
