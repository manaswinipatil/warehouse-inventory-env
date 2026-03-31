import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from models import ActionType, Direction, EnvironmentState, ItemModel, RewardInfo, RobotStateModel


class WarehouseInventoryEnv:
    """Warehouse inventory management environment with dynamic orders."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.grid_size = self.config.get("grid_size", (10, 10))
        self.max_steps = self.config.get("max_steps", 200)
        self.enable_orders = self.config.get("enable_orders", True)

        self._setup_spaces()

        self.env_state: Optional[EnvironmentState] = None
        self.step_count = 0
        self.task_id = 0
        self.charging_station = (0, 0)
        self.vision_radius = 5  # Partial observability: see only nearby grid

    def _setup_spaces(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "robot_position": spaces.Box(low=0, high=max(self.grid_size), shape=(2,), dtype=np.float32),
                "robot_battery": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "cargo_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
                "inventory_grid": spaces.Box(low=0, high=12, shape=self.grid_size, dtype=np.int32),
                "nearby_items": spaces.Box(low=0, high=self.grid_size[0], shape=(5, 3), dtype=np.float32),
                "task_progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "time_remaining": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "next_order_item": spaces.Box(low=-1, high=11, shape=(1,), dtype=np.int32),
            }
        )

        self.action_space = spaces.Dict(
            {
                "move_direction": spaces.Discrete(4),
                "action_type": spaces.Discrete(3),
                "target_item_id": spaces.Discrete(13),
            }
        )

    def reset(self, task_id: int = 0, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.task_id = task_id
        self.step_count = 0
        
        # Set max_steps based on task difficulty
        task_steps = {0: 50, 1: 100, 2: 200, 3: 250, 4: 300}
        self.max_steps = task_steps.get(task_id, 200)
        
        # Adjust vision based on task difficulty
        vision_by_task = {0: 10, 1: 8, 2: 6, 3: 5, 4: 4}
        self.vision_radius = vision_by_task.get(task_id, 5)

        robot = RobotStateModel(position=[0.0, 0.0], battery=1.0, cargo=[], max_cargo=5)
        items = self._generate_items(task_id)
        obstacles = self._generate_obstacles(task_id)
        dropoff_zone = (self.grid_size[0] - 1, self.grid_size[1] - 1)

        order_sequence: List[int] = []
        if self.enable_orders and task_id >= 1:
            available_items = [item.id for item in items]
            order_sequence = random.sample(available_items, min(len(available_items), 5))

        self.env_state = EnvironmentState(
            robot=robot,
            items=items,
            obstacles=obstacles,
            dropoff_zone=dropoff_zone,
            charging_station=self.charging_station,
            step_count=0,
            task_progress=0.0,
            total_reward=0.0,
            task_id=task_id,
            order_sequence=order_sequence,
            next_order_index=0,
        )
        return self._get_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.env_state is None:
            raise RuntimeError("Environment must be reset before stepping")

        self.step_count += 1
        self.env_state.step_count = self.step_count

        move_dir = int(action["move_direction"])
        action_type = int(action["action_type"])
        target_item_id = int(action["target_item_id"]) - 1

        reward_breakdown = RewardInfo()
        info: Dict[str, Any] = {
            "item_collected": False,
            "item_delivered": False,
            "item_id": None,
            "item_value": 0.0,
            "delivered_item_ids": [],
            "delivered_item_values": [],
            "distance_to_target": 0.0,
            "reward_breakdown": {},
            "order_completed": False,
        }

        if action_type == ActionType.MOVE:
            reward_breakdown.movement_penalty = self._move_robot(move_dir)
        elif action_type == ActionType.PICKUP:
            reward_breakdown.pickup_reward = self._pickup_item(target_item_id, info)
        elif action_type == ActionType.DROPOFF:
            reward_breakdown.dropoff_reward = self._dropoff_items(info)

        robot_pos_tuple = tuple(int(x) for x in self.env_state.robot.position)
        # Real warehouses often allow quick opportunity charging at the dispatch dock.
        if robot_pos_tuple == self.charging_station or robot_pos_tuple == self.env_state.dropoff_zone:
            self.env_state.robot.battery = min(1.0, self.env_state.robot.battery + 0.1)
            reward_breakdown.battery_cost = 0.1
        else:
            # Weight-based battery drain: heavier cargo drains faster
            cargo_weight = sum(
                item.weight for item in self.env_state.items 
                if item.id in self.env_state.robot.cargo
            )
            base_battery_cost = 0.015 * (1.0 + 0.5 * cargo_weight)
            self.env_state.robot.battery = max(0.0, self.env_state.robot.battery - base_battery_cost)
            reward_breakdown.battery_cost = -base_battery_cost * 0.5

        if self.env_state.robot.battery <= 0:
            done = True
            info["termination_reason"] = "battery_depleted"
            reward_breakdown.total = (
                reward_breakdown.movement_penalty
                + reward_breakdown.battery_cost
                + reward_breakdown.pickup_reward
                + reward_breakdown.dropoff_reward
                + reward_breakdown.progress_reward
                + reward_breakdown.efficiency_bonus
            )
            info["reward_breakdown"] = reward_breakdown.model_dump()
            self.env_state.total_reward += reward_breakdown.total
            return self._get_observation(), reward_breakdown.total, done, info

        self._update_task_progress()
        reward_breakdown.progress_reward = self._get_progress_reward()

        completed_orders = self._check_order_completion(info)
        if completed_orders > 0:
            reward_breakdown.efficiency_bonus += 2.0 * completed_orders
            info["order_completed"] = True
            info["orders_completed_in_step"] = completed_orders

        if self._is_task_complete():
            efficiency_bonus = 10.0 - (0.05 * self.step_count)
            reward_breakdown.efficiency_bonus += max(0.0, efficiency_bonus)
            info["task_completed"] = True

        reward_breakdown.total = (
            reward_breakdown.movement_penalty
            + reward_breakdown.battery_cost
            + reward_breakdown.pickup_reward
            + reward_breakdown.dropoff_reward
            + reward_breakdown.progress_reward
            + reward_breakdown.efficiency_bonus
        )
        info["reward_breakdown"] = reward_breakdown.model_dump()

        done = self._is_task_complete() or self.step_count >= self.max_steps
        if self.step_count >= self.max_steps:
            info["termination_reason"] = "max_steps_exceeded"

        self.env_state.total_reward += reward_breakdown.total
        return self._get_observation(), reward_breakdown.total, done, info

    def _get_observation(self) -> Dict[str, Any]:
        if self.env_state is None:
            return {}

        # Partial observability: only see items within vision radius
        inventory_grid = np.zeros(self.grid_size, dtype=np.int32)
        robot_x, robot_y = int(self.env_state.robot.position[0]), int(self.env_state.robot.position[1])
        
        for item in self.env_state.items:
            if not item.collected:
                x, y = item.position
                # Only show items within vision radius
                if abs(x - robot_x) <= self.vision_radius and abs(y - robot_y) <= self.vision_radius:
                    inventory_grid[x, y] = item.id + 1

        robot_pos = np.array(self.env_state.robot.position)
        nearby: List[Tuple[float, int, int, int]] = []
        for item in self.env_state.items:
            if not item.collected:
                dist = float(np.linalg.norm(np.array(item.position) - robot_pos))
                nearby.append((dist, item.position[0], item.position[1], item.id))

        nearby.sort(key=lambda entry: entry[0])
        nearby_items = np.zeros((5, 3), dtype=np.float32)
        for i, (_, x, y, item_id) in enumerate(nearby[:5]):
            nearby_items[i] = [x, y, item_id]

        next_order = -1
        if self.env_state.order_sequence and self.env_state.next_order_index < len(self.env_state.order_sequence):
            next_order = self.env_state.order_sequence[self.env_state.next_order_index]

        return {
            "robot_position": np.array(self.env_state.robot.position, dtype=np.float32),
            "robot_battery": np.array([self.env_state.robot.battery], dtype=np.float32),
            "cargo_count": np.array([self.env_state.robot.cargo_count()], dtype=np.int32),
            "inventory_grid": inventory_grid,
            "nearby_items": nearby_items,
            "task_progress": np.array([self.env_state.task_progress], dtype=np.float32),
            "time_remaining": np.array([1.0 - self.step_count / self.max_steps], dtype=np.float32),
            "next_order_item": np.array([next_order], dtype=np.int32),
        }

    def _move_robot(self, direction: int) -> float:
        if self.env_state is None:
            return -1.0

        new_pos = self.env_state.robot.position.copy()
        if direction == Direction.NORTH:
            new_pos[0] = max(0, new_pos[0] - 1)
        elif direction == Direction.EAST:
            new_pos[1] = min(self.grid_size[1] - 1, new_pos[1] + 1)
        elif direction == Direction.SOUTH:
            new_pos[0] = min(self.grid_size[0] - 1, new_pos[0] + 1)
        elif direction == Direction.WEST:
            new_pos[1] = max(0, new_pos[1] - 1)

        if tuple(int(v) for v in new_pos) in self.env_state.obstacles:
            return -2.0

        self.env_state.robot.position = new_pos
        return -0.03

    def _pickup_item(self, item_id: int, info: Dict[str, Any]) -> float:
        if self.env_state is None:
            return -1.0

        if not self.env_state.robot.can_pickup():
            return -1.0

        if item_id < 0:
            return -0.5

        robot_pos = tuple(int(x) for x in self.env_state.robot.position)
        for item in self.env_state.items:
            if not item.collected and item.id == item_id and item.position == robot_pos:
                if self.env_state.order_sequence and self.env_state.next_order_index < len(self.env_state.order_sequence):
                    expected = self.env_state.order_sequence[self.env_state.next_order_index]
                    if item.id != expected:
                        return -0.5

                if self.env_state.robot.add_cargo(item.id):
                    item.collected = True
                    info["item_collected"] = True
                    info["item_id"] = item.id
                    info["item_value"] = item.value
                    return float(item.value)
                return -1.0
        return -0.5

    def _dropoff_items(self, info: Dict[str, Any]) -> float:
        if self.env_state is None:
            return -1.0

        robot_pos = tuple(int(x) for x in self.env_state.robot.position)
        if robot_pos != self.env_state.dropoff_zone:
            return -1.0

        if not self.env_state.robot.cargo:
            return -0.5

        reward = 0.0
        delivered_ids: List[int] = []
        delivered_values: List[float] = []

        for item_id in list(self.env_state.robot.cargo):
            for item in self.env_state.items:
                if item.id == item_id:
                    item.delivered = True
                    delivered_ids.append(item_id)
                    delivered_values.append(float(item.value))
                    reward += float(item.value) * 2.0
                    break

        self.env_state.robot.cargo.clear()
        info["item_delivered"] = bool(delivered_ids)
        info["item_id"] = delivered_ids[-1] if delivered_ids else None
        info["item_value"] = delivered_values[-1] if delivered_values else 0.0
        info["delivered_item_ids"] = delivered_ids
        info["delivered_item_values"] = delivered_values
        return reward

    def _check_order_completion(self, info: Dict[str, Any]) -> int:
        if self.env_state is None or not self.env_state.order_sequence:
            return 0

        delivered_ids = info.get("delivered_item_ids", [])
        completed = 0

        for delivered_id in delivered_ids:
            if self.env_state.next_order_index >= len(self.env_state.order_sequence):
                break
            expected = self.env_state.order_sequence[self.env_state.next_order_index]
            if delivered_id == expected:
                self.env_state.next_order_index += 1
                completed += 1
        return completed

    def _generate_items(self, task_id: int) -> List[ItemModel]:
        items: List[ItemModel] = []

        if task_id == 0:
            # Task 0: Easy - stochastic 3 items
            num_items = 3
            for i in range(num_items):
                pos = (
                    random.randint(1, self.grid_size[0] - 2),
                    random.randint(1, self.grid_size[1] - 2),
                )
                items.append(
                    ItemModel(
                        id=i,
                        position=pos,
                        item_type=f"type_{i % 3}",
                        weight=random.uniform(0.7, 1.0),
                        value=1.0,
                    )
                )
        elif task_id == 1:
            # Task 1: Medium - 5 items with varied values
            for i in range(5):
                pos = (
                    random.randint(0, self.grid_size[0] - 1),
                    random.randint(0, self.grid_size[1] - 1),
                )
                items.append(
                    ItemModel(
                        id=i,
                        position=pos,
                        item_type=f"type_{i % 3}",
                        weight=random.uniform(0.8, 1.2),
                        value=random.uniform(0.5, 1.5),
                    )
                )
        elif task_id == 2:
            # Task 2: Hard - 8 items, some at corners
            for i in range(8):
                if i < 4:
                    pos = (
                        random.randint(1, self.grid_size[0] - 2),
                        random.randint(1, self.grid_size[1] - 2),
                    )
                else:
                    corners = [
                        (0, 0),
                        (0, self.grid_size[1] - 1),
                        (self.grid_size[0] - 1, 0),
                        (self.grid_size[0] - 1, self.grid_size[1] - 1),
                    ]
                    pos = corners[i - 4]

                items.append(
                    ItemModel(
                        id=i,
                        position=pos,
                        item_type=f"type_{i % 3}",
                        weight=random.uniform(0.9, 1.3),
                        value=random.uniform(0.8, 2.0),
                    )
                )
        elif task_id == 3:
            # Task 3: Very Hard - 10 items, scattered, heavy weights
            for i in range(10):
                pos = (
                    random.randint(0, self.grid_size[0] - 1),
                    random.randint(0, self.grid_size[1] - 1),
                )
                items.append(
                    ItemModel(
                        id=i,
                        position=pos,
                        item_type=f"type_{i % 4}",
                        weight=random.uniform(1.0, 1.5),
                        value=random.uniform(1.0, 2.5),
                    )
                )
        else:  # task_id == 4
            # Task 4: Extreme - 12 items, clustered, very heavy
            for i in range(12):
                # Create cluster pattern
                cluster_center = (
                    random.randint(2, self.grid_size[0] - 3),
                    random.randint(2, self.grid_size[1] - 3),
                )
                offset_x = random.randint(-2, 2)
                offset_y = random.randint(-2, 2)
                pos = (
                    max(0, min(self.grid_size[0] - 1, cluster_center[0] + offset_x)),
                    max(0, min(self.grid_size[1] - 1, cluster_center[1] + offset_y)),
                )
                items.append(
                    ItemModel(
                        id=i,
                        position=pos,
                        item_type=f"type_{i % 5}",
                        weight=random.uniform(1.2, 1.8),
                        value=random.uniform(1.5, 3.0),
                    )
                )
        return items

    def _generate_obstacles(self, task_id: int) -> List[Tuple[int, int]]:
        if task_id < 2:
            return []

        obstacles: List[Tuple[int, int]] = []
        
        if task_id == 2:
            # Task 2: Moderate obstacles - structured wall
            for i in range(3, 7):
                obstacles.append((i, 4))
                obstacles.append((i, 5))
            for _ in range(5):
                obstacles.append((random.randint(1, 8), random.randint(1, 8)))
                
        elif task_id == 3:
            # Task 3: Heavy obstacles - multiple walls and scattered blocks
            # Vertical wall
            for i in range(2, 8):
                obstacles.append((i, 3))
            # Horizontal wall
            for j in range(2, 8):
                obstacles.append((4, j))
            # Random blocks
            for _ in range(10):
                obstacles.append((random.randint(1, 8), random.randint(1, 8)))
                
        else:  # task_id == 4
            # Task 4: Extreme obstacles - maze-like
            # Multiple walls
            for i in range(1, 9):
                obstacles.append((i, 3))
                obstacles.append((i, 6))
            for j in range(1, 9):
                obstacles.append((3, j))
                obstacles.append((6, j))
            # Dense random obstacles
            for _ in range(15):
                obstacles.append((random.randint(1, 8), random.randint(1, 8)))

        # Keep special cells always free
        blocked = {self.charging_station, (self.grid_size[0] - 1, self.grid_size[1] - 1)}
        obstacles = [obs for obs in obstacles if obs not in blocked]
        return list(dict.fromkeys(obstacles))

    def _update_task_progress(self) -> None:
        if self.env_state is None:
            return

        total = len(self.env_state.items)
        if total == 0:
            self.env_state.task_progress = 1.0
            return

        delivered = len([item for item in self.env_state.items if item.delivered])
        self.env_state.task_progress = delivered / total

    def _is_task_complete(self) -> bool:
        if self.env_state is None:
            return False

        all_delivered = all(item.delivered for item in self.env_state.items)
        robot_at_origin = tuple(int(v) for v in self.env_state.robot.position) == (0, 0)
        
        # Tasks 2, 3, 4 require return to origin
        if self.task_id >= 2:
            return all_delivered and robot_at_origin
        return all_delivered

    def _get_progress_reward(self) -> float:
        if self.env_state is None:
            return 0.0

        delivered = sum(1 for item in self.env_state.items if item.delivered)
        total = len(self.env_state.items)
        return (delivered / total) * 0.15 if total > 0 else 0.0

    def state(self) -> Dict[str, Any]:
        if self.env_state is None:
            return {}
        return self.env_state.model_dump()
