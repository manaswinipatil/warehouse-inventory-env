from typing import Any, Dict, List, Set, Tuple

import numpy as np


def _strict_score(value: float, epsilon: float = 1e-3) -> float:
    return float(min(1.0 - epsilon, max(epsilon, float(value))))


class TaskGrader:
    """Base class for deterministic task graders."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        raise NotImplementedError


class EasyNavigationGrader(TaskGrader):
    """Deterministic grader for easy navigation task."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return _strict_score(0.0)

        collected_items: Set[int] = set()
        for step in trajectory:
            info = step.get("info", {})
            if info.get("item_collected", False):
                item_id = info.get("item_id")
                if item_id is not None:
                    collected_items.add(int(item_id))

        return _strict_score(len(collected_items) / 3.0)


class MediumPickupGrader(TaskGrader):
    """Deterministic grader for medium pickup task."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return _strict_score(0.0)

        collected_items: Set[int] = set()
        delivered_items: Set[int] = set()

        for step in trajectory:
            info = step.get("info", {})
            if info.get("item_collected", False):
                item_id = info.get("item_id")
                if item_id is not None:
                    collected_items.add(int(item_id))

            for delivered_id in info.get("delivered_item_ids", []):
                delivered_items.add(int(delivered_id))

        battery_levels = [float(step["observation"]["robot_battery"][0]) for step in trajectory]
        avg_battery = float(np.mean(battery_levels)) if battery_levels else 0.0

        collection_score = len(collected_items) / 5.0
        delivery_score = len(delivered_items) / 5.0
        battery_score = avg_battery

        score = 0.4 * collection_score + 0.4 * delivery_score + 0.2 * battery_score
        return _strict_score(score)


class HardInventoryGrader(TaskGrader):
    """Deterministic grader for hard inventory task."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return _strict_score(0.0)

        delivered_items: Set[int] = set()
        total_value = 0.0
        final_position: Tuple[float, float] = (0.0, 0.0)

        for idx, step in enumerate(trajectory):
            info = step.get("info", {})

            delivered_ids = info.get("delivered_item_ids", [])
            delivered_values = info.get("delivered_item_values", [])
            for delivered_id in delivered_ids:
                delivered_items.add(int(delivered_id))
            total_value += sum(float(v) for v in delivered_values)

            if idx == len(trajectory) - 1:
                obs = step.get("next_observation", step.get("observation", {}))
                pos = obs.get("robot_position", [0.0, 0.0])
                final_position = (float(pos[0]), float(pos[1]))

        completion_score = len(delivered_items) / 8.0
        value_score = min(1.0, total_value / 16.0)
        return_bonus = 1.0 if final_position == (0.0, 0.0) else 0.0

        score = 0.5 * completion_score + 0.3 * value_score + 0.2 * return_bonus
        return _strict_score(score)


class VeryHardInventoryGrader(TaskGrader):
    """Deterministic grader for very hard inventory task (10 items)."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return _strict_score(0.0)

        delivered_items: Set[int] = set()
        total_value = 0.0
        final_position: Tuple[float, float] = (0.0, 0.0)
        episodes_to_complete = len(trajectory)

        for idx, step in enumerate(trajectory):
            info = step.get("info", {})

            delivered_ids = info.get("delivered_item_ids", [])
            delivered_values = info.get("delivered_item_values", [])
            for delivered_id in delivered_ids:
                delivered_items.add(int(delivered_id))
            total_value += sum(float(v) for v in delivered_values)

            if idx == len(trajectory) - 1:
                obs = step.get("next_observation", step.get("observation", {}))
                pos = obs.get("robot_position", [0.0, 0.0])
                final_position = (float(pos[0]), float(pos[1]))

        completion_score = len(delivered_items) / 10.0
        value_score = min(1.0, total_value / 25.0)
        return_bonus = 1.0 if final_position == (0.0, 0.0) else 0.0
        efficiency_penalty = min(0.3, episodes_to_complete / 250.0)

        score = 0.45 * completion_score + 0.35 * value_score + 0.2 * return_bonus - efficiency_penalty
        return _strict_score(score)


class ExtremeInventoryGrader(TaskGrader):
    """Deterministic grader for extreme inventory task (12 items, clustered)."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return _strict_score(0.0)

        delivered_items: Set[int] = set()
        total_value = 0.0
        final_position: Tuple[float, float] = (0.0, 0.0)
        total_steps = len(trajectory)

        for idx, step in enumerate(trajectory):
            info = step.get("info", {})

            delivered_ids = info.get("delivered_item_ids", [])
            delivered_values = info.get("delivered_item_values", [])
            for delivered_id in delivered_ids:
                delivered_items.add(int(delivered_id))
            total_value += sum(float(v) for v in delivered_values)

            if idx == len(trajectory) - 1:
                obs = step.get("next_observation", step.get("observation", {}))
                pos = obs.get("robot_position", [0.0, 0.0])
                final_position = (float(pos[0]), float(pos[1]))

        completion_score = len(delivered_items) / 12.0
        value_score = min(1.0, total_value / 36.0)
        return_bonus = 1.0 if final_position == (0.0, 0.0) else 0.0
        efficiency_bonus = max(0.0, 1.0 - total_steps / 300.0)

        score = 0.4 * completion_score + 0.35 * value_score + 0.15 * return_bonus + 0.1 * efficiency_bonus
        return _strict_score(score)
