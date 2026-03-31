from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ActionType(IntEnum):
    MOVE = 0
    PICKUP = 1
    DROPOFF = 2


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class ObservationModel(BaseModel):
    """Complete observation space model."""

    robot_position: List[float] = Field(..., min_length=2, max_length=2)
    robot_battery: List[float] = Field(..., min_length=1, max_length=1)
    cargo_count: List[int] = Field(..., min_length=1, max_length=1)
    inventory_grid: List[List[int]] = Field(..., min_length=10, max_length=10)
    nearby_items: List[List[float]] = Field(..., min_length=5, max_length=5)
    task_progress: List[float] = Field(..., min_length=1, max_length=1)
    time_remaining: List[float] = Field(..., min_length=1, max_length=1)
    next_order_item: List[int] = Field(..., min_length=1, max_length=1)

    @field_validator("robot_position")
    @classmethod
    def validate_position(cls, value: List[float]) -> List[float]:
        if not (0 <= value[0] <= 9 and 0 <= value[1] <= 9):
            raise ValueError(f"Position {value} out of bounds")
        return value

    @field_validator("robot_battery")
    @classmethod
    def validate_battery(cls, value: List[float]) -> List[float]:
        if not (0 <= value[0] <= 1):
            raise ValueError(f"Battery {value[0]} out of range")
        return value

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {np.ndarray: lambda arr: arr.tolist()},
    }


class ActionModel(BaseModel):
    """Complete action space model."""

    move_direction: int = Field(..., ge=0, le=3)
    action_type: int = Field(..., ge=0, le=2)
    target_item_id: int = Field(..., ge=0, le=12)

    @field_validator("move_direction")
    @classmethod
    def validate_direction(cls, value: int) -> int:
        if value not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid direction {value}")
        return value

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, value: int) -> int:
        if value not in [0, 1, 2]:
            raise ValueError(f"Invalid action type {value}")
        return value


class RewardInfo(BaseModel):
    """Structured reward breakdown."""

    movement_penalty: float = 0.0
    battery_cost: float = 0.0
    pickup_reward: float = 0.0
    dropoff_reward: float = 0.0
    progress_reward: float = 0.0
    efficiency_bonus: float = 0.0
    total: float = 0.0


class StepResult(BaseModel):
    """Complete step result."""

    observation: ObservationModel
    reward: float
    done: bool
    info: Dict[str, Any]


class RobotStateModel(BaseModel):
    """Internal robot state."""

    position: List[float]
    battery: float = Field(..., ge=0.0, le=1.0)
    cargo: List[int] = Field(default_factory=list)
    max_cargo: int = Field(5, ge=1, le=10)

    def cargo_count(self) -> int:
        return len(self.cargo)

    def can_pickup(self) -> bool:
        return self.cargo_count() < self.max_cargo

    def add_cargo(self, item_id: int) -> bool:
        if self.can_pickup():
            self.cargo.append(item_id)
            return True
        return False

    def remove_cargo(self, item_id: int) -> bool:
        if item_id in self.cargo:
            self.cargo.remove(item_id)
            return True
        return False


class ItemModel(BaseModel):
    """Warehouse item model."""

    id: int = Field(..., ge=0, lt=100)
    position: Tuple[int, int]
    item_type: str
    weight: float = Field(..., ge=0.0)
    value: float = Field(..., ge=0.0)
    collected: bool = False
    delivered: bool = False


class EnvironmentState(BaseModel):
    """Complete environment state."""

    robot: RobotStateModel
    items: List[ItemModel]
    obstacles: List[Tuple[int, int]]
    dropoff_zone: Tuple[int, int]
    charging_station: Tuple[int, int] = (0, 0)
    step_count: int = 0
    task_progress: float = 0.0
    total_reward: float = 0.0
    task_id: int = 0
    order_sequence: List[int] = Field(default_factory=list)
    next_order_index: int = 0


class GraderRequest(BaseModel):
    task_id: int = Field(..., ge=0, le=4)
    trajectory: List[Dict[str, Any]]


class BaselineScores(BaseModel):
    easy: float
    medium: float
    hard: float
    very_hard: float
    extreme: float
    overall: float


class TasksResponse(BaseModel):
    tasks: List[Dict[str, Any]]


class ResetResponse(BaseModel):
    task_id: int
    observation: Dict[str, Any]


class MetricsResponse(BaseModel):
    step_count: int
    total_reward: float
    battery: float
    task_progress: float
    cargo_count: int
    task_id: Optional[int] = None
