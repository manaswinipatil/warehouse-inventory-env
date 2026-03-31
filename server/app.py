from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment import WarehouseInventoryEnv
from inference import run_inference
from models import (
    ActionModel,
    BaselineScores,
    GraderRequest,
    MetricsResponse,
    ResetResponse,
    TasksResponse,
)
from tasks import EasyNavigationGrader, HardInventoryGrader, MediumPickupGrader, VeryHardInventoryGrader, ExtremeInventoryGrader


app = FastAPI(title="Warehouse Inventory Environment", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = WarehouseInventoryEnv()

TASKS = [
    {
        "id": 0,
        "name": "easy_navigation",
        "description": "Navigate and collect 3 randomly-placed items with partial observability",
        "max_steps": 50,
        "grader": "EasyNavigationGrader",
        "module": "tasks",
        "weight": 0.15,
    },
    {
        "id": 1,
        "name": "medium_pickup",
        "description": "Collect 5 items with dynamic orders, battery management, and partial observability",
        "max_steps": 100,
        "grader": "MediumPickupGrader",
        "module": "tasks",
        "weight": 0.2,
    },
    {
        "id": 2,
        "name": "hard_inventory",
        "description": "Full inventory management with 8 items, obstacles, orders, partial observability, and return",
        "max_steps": 200,
        "grader": "HardInventoryGrader",
        "module": "tasks",
        "weight": 0.25,
    },
    {
        "id": 3,
        "name": "very_hard_inventory",
        "description": "Extreme challenge: 10+ items, maze-like obstacles, heavy weights, partial observability, return to origin",
        "max_steps": 250,
        "grader": "VeryHardInventoryGrader",
        "module": "tasks",
        "weight": 0.2,
    },
    {
        "id": 4,
        "name": "extreme_inventory",
        "description": "Hardest: 12 clustered items, dense maze, heavy weights, strict time pressure, return to origin",
        "max_steps": 300,
        "grader": "ExtremeInventoryGrader",
        "module": "tasks",
        "weight": 0.2,
    },
]

GRADERS = {
    0: EasyNavigationGrader(),
    1: MediumPickupGrader(),
    2: HardInventoryGrader(),
    3: VeryHardInventoryGrader(),
    4: ExtremeInventoryGrader(),
}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "warehouse-inventory-env",
        "version": "1.0.0",
        "status": "ok",
    }


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    return TasksResponse(tasks=TASKS)


@app.post("/reset", response_model=ResetResponse)
def reset(task_id: int = 0, seed: int = 42) -> ResetResponse:
    if task_id not in [0, 1, 2, 3, 4]:
        raise HTTPException(status_code=400, detail="task_id must be 0-4")

    observation = env.reset(task_id=task_id, seed=seed)
    serializable_obs = _to_serializable(observation)
    return ResetResponse(task_id=task_id, observation=serializable_obs)


@app.post("/step")
def step(action: ActionModel) -> Dict[str, Any]:
    if env.env_state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    observation, reward, done, info = env.step(action.model_dump())
    return {
        "observation": _to_serializable(observation),
        "reward": float(reward),
        "done": bool(done),
        "info": _to_serializable(info),
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return _to_serializable(env.state())


@app.get("/baseline", response_model=BaselineScores)
def baseline(num_episodes: int = 3) -> BaselineScores:
    if num_episodes < 1:
        raise HTTPException(status_code=400, detail="num_episodes must be >= 1")

    easy = run_inference(task_id=0, num_episodes=num_episodes, verbose=False)
    medium = run_inference(task_id=1, num_episodes=num_episodes, verbose=False)
    hard = run_inference(task_id=2, num_episodes=num_episodes, verbose=False)
    very_hard = run_inference(task_id=3, num_episodes=num_episodes, verbose=False)
    extreme = run_inference(task_id=4, num_episodes=num_episodes, verbose=False)
    overall = (easy + medium + hard + very_hard + extreme) / 5.0

    return BaselineScores(
        easy=easy,
        medium=medium,
        hard=hard,
        very_hard=very_hard,
        extreme=extreme,
        overall=overall,
    )


@app.post("/grader")
def grader(request: GraderRequest) -> Dict[str, float]:
    grader_impl = GRADERS.get(request.task_id)
    if grader_impl is None:
        raise HTTPException(status_code=400, detail="Unsupported task_id")

    score = grader_impl.grade(request.trajectory)
    return {"task_id": request.task_id, "score": float(score)}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    if env.env_state is None:
        return MetricsResponse(
            step_count=0,
            total_reward=0.0,
            battery=1.0,
            task_progress=0.0,
            cargo_count=0,
            task_id=None,
        )

    return MetricsResponse(
        step_count=int(env.step_count),
        total_reward=float(env.env_state.total_reward),
        battery=float(env.env_state.robot.battery),
        task_progress=float(env.env_state.task_progress),
        cargo_count=int(env.env_state.robot.cargo_count()),
        task_id=int(env.task_id),
    )


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def main() -> None:
    """Entry point for the warehouse inventory server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
