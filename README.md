# Warehouse Inventory OpenEnv

A real-world warehouse inventory management environment designed for agent training and evaluation with OpenEnv-style APIs. Features stochastic item spawning, dynamic obstacles, partial observability, weight-based battery dynamics, and 5 progressive difficulty tasks.

## Key Features

**Stochastic Environment:**
- All items spawn randomly per seed (not fixed positions)
- Dynamic obstacles that vary by task difficulty
- Partial observability: agents see only within a vision radius

**Realistic Mechanics:**
- Weight-based battery drain: heavier cargo → faster drain
- Partial charge at dock/charging station
- Origin-return requirement for hard tasks
- Dense, nuanced reward signals (not sparse)

**5 Progressive Tasks:**
1. **Easy Navigation** (3 items, no obstacles)
2. **Medium Pickup** (5 items, dynamic orders)
3. **Hard Inventory** (8 items, obstacles, return-to-origin)
4. **Very Hard** (10 items, maze obstacles, heavy weights)
5. **Extreme** (12 clustered items, dense maze, time pressure)

## Why This Environment

Modern fulfillment centers rely on autonomous robots to:
- navigate storage aisles (with partially known layout),
- collect ordered items (with stochastic positioning),
- manage battery constraints under varying loads,
- deliver cargo to a drop-off station,
- satisfy dynamic item order sequences.

This environment models those constraints with dense rewards, safety penalties, and multi-task grading that challenges frontier LLMs and RL agents.

## Project Structure

- `app.py`: FastAPI service exposing reset/step/state and evaluation endpoints
- `environment.py`: Core warehouse simulation with stochastic spawning, dynamics, and partial obs
- `models.py`: Typed Pydantic models for observations, actions, and responses
- `tasks.py`: Deterministic task graders with normalized 0.0-1.0 scores (5 tasks)
- `inference.py`: Reproducible baseline inference runner using heuristic agent
- `openenv.yaml`: Environment metadata and task specification
- `Dockerfile`: Container runtime for Hugging Face Spaces
- `test_api.py`: Endpoint smoke tests

## Observation Space

Observation is a dictionary with:
- `robot_position`: float[2], robot grid coordinate [x, y]
- `robot_battery`: float[1], normalized battery in [0, 1]
- `cargo_count`: int[1], number of items currently carried
- `inventory_grid`: int[10, 10], **partial map**: only items within vision_radius visible (increases difficulty with task)
- `nearby_items`: float[5, 3], nearest item summaries [x, y, item_id]
- `task_progress`: float[1], normalized delivered progress in [0, 1]
- `time_remaining`: float[1], normalized remaining steps in [0, 1]
- `next_order_item`: int[1], required next item ID, or -1

## Action Space

Action is a dictionary with:
- `move_direction`: int in [0, 3]
  - 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST
- `action_type`: int in [0, 2]
  - 0 = MOVE, 1 = PICKUP, 2 = DROPOFF
- `target_item_id`: int in [0, 9]
  - 0 represents no target for movement/dropoff actions
  - 1..9 maps to environment item IDs 0..8 for pickup

## Reward Design

Each step returns a dense reward composed of:
- `movement_penalty`: discourages unnecessary movement/collisions
- `battery_cost`: **weight-based**: heavier cargo = higher drain, rewards charging station and dock
- `pickup_reward`: reward from successful item pickups
- `dropoff_reward`: larger reward from successful deliveries
- `progress_reward`: partial-progress signal based on delivered ratio
- `efficiency_bonus`: completion and order-efficiency bonuses

This provides non-sparse guidance while discouraging looping and invalid actions.

## Tasks and Graders

Five deterministic tasks with increasing difficulty:

| Task | Items | Max Steps | Vision | Obstacles | Return? | Grader |
|------|-------|-----------|--------|-----------|---------|--------|
| 0. Easy Navigation | 3 | 50 | 10 | No | No | Collection count / 3.0 |
| 1. Medium Pickup | 5 | 100 | 8 | No | No | Mix of collection, delivery, battery efficiency |
| 2. Hard Inventory | 8 | 200 | 6 | Moderate | Yes | Completion + value + return bonus |
| 3. Very Hard | 10 | 250 | 5 | Heavy maze | Yes | 10-item completion (stricter) |
| 4. Extreme | 12 | 300 | 4 | Dense maze | Yes | 12-item clustering challenge |

All graders return bounded scores in [0.0, 1.0] with deterministic, reproducible evaluation.

## API Endpoints

- `GET /` — Health check
- `GET /tasks` — List all 5 tasks and metadata
- `POST /reset?task_id=<0-4>&seed=<int>` — Initialize task
- `POST /step` — Execute one action
- `GET /state` — Retrieve current environment state
- `GET /baseline?num_episodes=<int>` — Run heuristic baseline on tasks 0-2
- `POST /grader` — Grade a trajectory
- `GET /metrics` — Current episode metrics

## Setup

### Local Python

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start API:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

3. Validate endpoints:

```bash
python test_api.py
```

### Baseline Inference

Run all tasks (quick 2-episode baseline):

```bash
python inference.py
```

Run programmatically:

```python
from inference import run_inference
print(run_inference(task_id=0, num_episodes=5))
```

## Environment Variables

The baseline script supports the required variables:
- `OPENAI_API_KEY` — OpenAI API key for optional LLM-based agent
- `API_BASE_URL` — Custom LLM endpoint
- `MODEL_NAME` — Model identifier (default: gpt-4o-mini)
- `HF_TOKEN` — Hugging Face token (for deployment)

If OpenAI client is unavailable, the script uses a deterministic heuristic baseline.

## Docker

Build and run locally:

```bash
docker build -t warehouse-inventory-env .
docker run -p 7860:7860 warehouse-inventory-env
```

## Hugging Face Spaces Deployment

- Use Docker Space type
- Ensure container exposes port 7860
- Provide required secrets in Space settings:
  - `OPENAI_API_KEY`
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`

## Reproducibility

- `PYTHONHASHSEED` is set in inference.py
- numpy and random seeds are fixed
- run_inference uses deterministic per-episode seeding
- Stochastic spawning is seed-dependent (same seed = same layout)

## Baseline Scores

Current snapshot (2 episodes each, heuristic agent):
- **Easy Navigation**: 1.000
- **Medium Pickup**: 0.655
- **Hard Inventory**: 0.200
- **Very Hard**: 0.000 (maze navigation challenge)
- **Extreme**: 0.150 (very heavy weights, time pressure)

**Overall**: 0.401

Note: Lower hard-task scores show the environment is genuinely challenging. Real frontier LLMs and advanced RL agents should achieve much higher scores through learning, planning, and partial-observability reasoning.

## Innovation Highlights

✅ **Stochastic Spawning** — Items and obstacles change every episode (not fixed)
✅ **Partial Observability** — Vision radius decreases with task difficulty (realistic localization constraint)
✅ **Weight-Based Dynamics** — Heavier items drain battery faster (realistic physics)
✅ **5-Task Ladder** — Clear progression from easy to extreme (curriculum for eval)
✅ **Return-to-Origin** — Tasks 2-4 require safe return (realistic constraint in logistics)

## Future Enhancements

- Multi-robot coordination
- Dynamic environment changes (obstacles appear/disappear)
- Sensor noise / imperfect observations
- Energy constraints with active-learning strategies
