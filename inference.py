import json
import os
import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from environment import WarehouseInventoryEnv
from models import Direction
from tasks import EasyNavigationGrader, HardInventoryGrader, MediumPickupGrader, VeryHardInventoryGrader, ExtremeInventoryGrader


os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

# Accept HF_TOKEN as a fallback API key for hosted runtimes that expose only this variable.
if not OPENAI_API_KEY and HF_TOKEN:
    OPENAI_API_KEY = HF_TOKEN


def emit_start(task: str) -> None:
    # Required by evaluator: structured START line on stdout.
    print(f"[START] task={task}", flush=True)


def emit_step(step: int, reward: float) -> None:
    # Required by evaluator: structured STEP line with stable field names/order.
    print(f"[STEP] step={int(step)} reward={float(reward):.6f}", flush=True)


def emit_end(task: str, score: float, steps: int) -> None:
    # Required by evaluator: structured END line on stdout.
    print(
        f"[END] task={task} score={float(score):.6f} steps={int(steps)}",
        flush=True,
    )


class InferenceAgent:
    """Inference agent for environment evaluation."""

    def __init__(self, env: WarehouseInventoryEnv, use_heuristic: bool = True):
        self.env = env
        self.use_heuristic = use_heuristic

        self.openai_client = None
        if OPENAI_API_KEY and not use_heuristic:
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
            except Exception:
                self.openai_client = None

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.openai_client and not self.use_heuristic:
            openai_action = self._openai_action(observation)
            if openai_action:
                return openai_action

        return self._heuristic_action(observation)

    def _heuristic_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        robot_pos_arr = observation["robot_position"]
        robot_pos = (int(robot_pos_arr[0]), int(robot_pos_arr[1]))
        cargo_count = int(observation["cargo_count"][0])
        next_order = int(observation.get("next_order_item", [-1])[0])
        battery = float(observation["robot_battery"][0])
        max_cargo = int(self.env.env_state.robot.max_cargo)
        remaining_steps = int(self.env.max_steps - self.env.step_count)

        dropoff_zone = (
            int(self.env.env_state.dropoff_zone[0]),
            int(self.env.env_state.dropoff_zone[1]),
        )
        charging_station = tuple(int(v) for v in self.env.charging_station)
        robot_cargo = list(self.env.env_state.robot.cargo)
        obstacles = self._obstacles_set()
        available_items = self._available_items(observation)
        carrying_required_order = next_order >= 0 and next_order in robot_cargo
        waiting_for_order_delivery = (
            next_order >= 0 and next_order not in available_items and carrying_required_order
        )

        # Always unload immediately once at drop-off.
        if robot_pos == dropoff_zone and cargo_count > 0:
            return {"move_direction": 0, "action_type": 2, "target_item_id": 0}

        collect_target_id, collect_target_pos = self._choose_collect_target(
            robot_pos=robot_pos,
            next_order=next_order,
            available_items=available_items,
        )

        if self.env.task_id == 2 and collect_target_id is not None:
            hard_carry_limit = 2 if next_order >= 0 else 3
        elif self.env.task_id == 3 and collect_target_id is not None:
            hard_carry_limit = 2  # Very heavy items, carry less
        elif self.env.task_id == 4 and collect_target_id is not None:
            hard_carry_limit = 1  # Extreme weights, single item at a time
        else:
            hard_carry_limit = max_cargo

        # For hard tasks (2, 3, 4), plan early return to origin
        if self.env.task_id >= 2:
            steps_to_origin = self._path_length(robot_pos, charging_station, obstacles)
            budget_needed = steps_to_origin + 5  # Buffer for unexpected obstacles
            if remaining_steps <= budget_needed:
                move_dir = self._path_move(robot_pos, charging_station, obstacles)
                return self._move_action(move_dir)

        # Attempt pickup whenever standing on intended target.
        if (
            collect_target_id is not None
            and collect_target_pos is not None
            and robot_pos == collect_target_pos
            and cargo_count < max_cargo
        ):
            return {
                "move_direction": 0,
                "action_type": 1,
                "target_item_id": collect_target_id + 1,
            }

        must_charge = battery <= (0.28 if cargo_count == 0 else 0.35)
        
        # For harder tasks, charge more aggressively
        if self.env.task_id >= 3:
            must_charge = battery <= (0.4 if cargo_count == 0 else 0.5)
        elif self.env.task_id == 2:
            must_charge = battery <= (0.3 if cargo_count == 0 else 0.4)
        must_dropoff = (
            cargo_count >= hard_carry_limit
            or waiting_for_order_delivery
            or carrying_required_order
            or (cargo_count > 0 and collect_target_id is None)
            or (cargo_count > 0 and battery <= 0.42)
            or (cargo_count > 0 and remaining_steps <= 20)
        )
        
        # For very hard tasks, drop off more frequently
        if self.env.task_id == 3 and cargo_count > 0 and battery <= 0.5:
            must_dropoff = True
        if self.env.task_id == 4 and cargo_count > 0 and battery <= 0.6:
            must_dropoff = True

        if must_charge and robot_pos != charging_station:
            move_dir = self._path_move(robot_pos, charging_station, obstacles)
            return self._move_action(move_dir)

        if must_charge and robot_pos == charging_station and battery < 0.95:
            return self._move_action(Direction.NORTH)

        if must_dropoff:
            move_dir = self._path_move(robot_pos, dropoff_zone, obstacles)
            return self._move_action(move_dir)

        if collect_target_pos is not None:
            move_dir = self._path_move(robot_pos, collect_target_pos, obstacles)
            return self._move_action(move_dir)

        # No items left to collect; route toward final objective.
        if self.env.task_id == 2:
            move_dir = self._path_move(robot_pos, charging_station, obstacles)
            return self._move_action(move_dir)

        move_dir = self._path_move(robot_pos, dropoff_zone, obstacles)
        return self._move_action(move_dir)

    def _move_action(self, move_direction: int) -> Dict[str, Any]:
        return {
            "move_direction": int(move_direction),
            "action_type": 0,
            "target_item_id": 0,
        }

    def _obstacles_set(self) -> Set[Tuple[int, int]]:
        return set((int(x), int(y)) for x, y in self.env.env_state.obstacles)

    def _available_items(self, observation: Dict[str, Any]) -> Dict[int, Tuple[int, int]]:
        grid = observation["inventory_grid"]
        found: Dict[int, Tuple[int, int]] = {}

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                value = int(grid[x, y])
                if value > 0:
                    found[value - 1] = (x, y)

        return found

    def _choose_collect_target(
        self,
        robot_pos: Tuple[int, int],
        next_order: int,
        available_items: Dict[int, Tuple[int, int]],
    ) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        if next_order >= 0 and next_order in available_items:
            return next_order, available_items[next_order]

        if not available_items:
            return None, None

        if self.env.task_id == 2:
            best_item_id = max(
                available_items,
                key=lambda item_id: self._hard_item_priority(robot_pos, item_id, available_items[item_id]),
            )
            return best_item_id, available_items[best_item_id]

        nearest_item_id = min(
            available_items,
            key=lambda item_id: abs(available_items[item_id][0] - robot_pos[0])
            + abs(available_items[item_id][1] - robot_pos[1]),
        )
        return nearest_item_id, available_items[nearest_item_id]

    def _hard_item_priority(
        self,
        robot_pos: Tuple[int, int],
        item_id: int,
        item_pos: Tuple[int, int],
    ) -> float:
        item_value = 1.0
        for item in self.env.env_state.items:
            if item.id == item_id:
                item_value = float(item.value)
                break

        dist = abs(item_pos[0] - robot_pos[0]) + abs(item_pos[1] - robot_pos[1])
        dropoff = tuple(int(v) for v in self.env.env_state.dropoff_zone)
        dock_dist = abs(item_pos[0] - dropoff[0]) + abs(item_pos[1] - dropoff[1])
        return item_value - 0.06 * dist - 0.03 * dock_dist

    def _path_move(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
    ) -> int:
        if start == goal:
            return Direction.NORTH

        max_x, max_y = self.env.grid_size
        queue: deque[Tuple[int, int]] = deque([start])
        parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        neighbors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        while queue:
            current = queue.popleft()
            if current == goal:
                break

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                nxt = (nx, ny)
                if nx < 0 or ny < 0 or nx >= max_x or ny >= max_y:
                    continue
                if nxt in obstacles or nxt in parents:
                    continue
                parents[nxt] = current
                queue.append(nxt)

        if goal not in parents:
            return self._greedy_fallback(start, goal, obstacles)

        step = goal
        while parents[step] is not None and parents[step] != start:
            step = parents[step]

        if parents[step] is None:
            return Direction.NORTH

        dx = step[0] - start[0]
        dy = step[1] - start[1]
        if dx < 0:
            return Direction.NORTH
        if dx > 0:
            return Direction.SOUTH
        if dy > 0:
            return Direction.EAST
        return Direction.WEST

    def _greedy_fallback(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
    ) -> int:
        candidates: List[Tuple[int, int, int]] = []
        directions = [
            (Direction.NORTH, (start[0] - 1, start[1])),
            (Direction.EAST, (start[0], start[1] + 1)),
            (Direction.SOUTH, (start[0] + 1, start[1])),
            (Direction.WEST, (start[0], start[1] - 1)),
        ]

        max_x, max_y = self.env.grid_size
        for direction, pos in directions:
            x, y = pos
            if x < 0 or y < 0 or x >= max_x or y >= max_y:
                continue
            if pos in obstacles:
                continue
            score = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            candidates.append((score, int(direction), random.randint(0, 1000)))

        if not candidates:
            return Direction.NORTH

        candidates.sort(key=lambda item: (item[0], item[2]))
        return candidates[0][1]

    def _path_length(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
    ) -> int:
        if start == goal:
            return 0

        max_x, max_y = self.env.grid_size
        queue: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        seen: Set[Tuple[int, int]] = {start}
        neighbors = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while queue:
            current, distance = queue.popleft()
            if current == goal:
                return distance

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                nxt = (nx, ny)
                if nx < 0 or ny < 0 or nx >= max_x or ny >= max_y:
                    continue
                if nxt in obstacles or nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, distance + 1))

        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def _openai_action(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            obs_str = (
                f"Robot at {observation['robot_position']}, "
                f"battery {observation['robot_battery'][0]:.2f}, "
                f"cargo {observation['cargo_count'][0]}, "
                f"next order: {observation.get('next_order_item', [-1])[0]}"
            )

            response = self.openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a warehouse robot. Return strict JSON with keys "
                            "move_direction (0-3), action_type (0-2), target_item_id (0-12)."
                        ),
                    },
                    {"role": "user", "content": f"State: {obs_str}"},
                ],
                max_tokens=50,
                temperature=0,
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return {
                "move_direction": int(parsed["move_direction"]),
                "action_type": int(parsed["action_type"]),
                "target_item_id": int(parsed["target_item_id"]),
            }
        except Exception:
            return None


def run_inference(
    task_id: int,
    num_episodes: int = 5,
    verbose: bool = True,
    use_heuristic: bool = True,
) -> float:
    env = WarehouseInventoryEnv()
    agent = InferenceAgent(env, use_heuristic=use_heuristic)

    graders = [EasyNavigationGrader(), MediumPickupGrader(), HardInventoryGrader(), VeryHardInventoryGrader(), ExtremeInventoryGrader()]
    total_scores: List[float] = []

    for episode in range(num_episodes):
        observation = env.reset(task_id=task_id, seed=episode)
        trajectory: List[Dict[str, Any]] = []
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(observation)
            next_obs, reward, done, info = env.step(action)

            trajectory.append(
                {
                    "observation": observation,
                    "action": action,
                    "reward": reward,
                    "next_observation": next_obs,
                    "info": info,
                }
            )

            observation = next_obs
            total_reward += float(reward)

        score = graders[task_id].grade(trajectory)
        total_scores.append(score)

        if verbose:
            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Score={score:.3f}")

    avg_score = float(np.mean(total_scores)) if total_scores else 0.0
    if verbose:
        print(f"\nAverage score for task {task_id}: {avg_score:.3f}")

    return avg_score


if __name__ == "__main__":
    scores: List[float] = []
    task_names = [
        "Easy Navigation",
        "Medium Pickup",
        "Hard Inventory",
        "Very Hard Inventory",
        "Extreme Inventory"
    ]

    for task_id in range(5):
        task_name = task_names[task_id]
        emit_start(task_name)
        score = run_inference(task_id, num_episodes=2, verbose=False)  # 2 episodes for speed
        emit_step(1, score)
        emit_end(task_name, score, 1)
        scores.append(score)
