# Chapter 13 Instructor Guide
## Reinforcement Learning for Robotics, Embodied AI, and Vision-Language-Action Systems

This repository is the instructor-facing companion for the Chapter 13 exercises. It provides:

- complete written solutions for all 18 exercises;
- a tested Gymnasium-style robotic reaching environment;
- Colab-ready notebooks for environment validation, behavior cloning, perturbation testing, offline-dataset design, VLA benchmark design, and method comparison;
- reproducible seeds, logging conventions, and lightweight CPU-compatible examples;
- instructor notes, expected results, grading rubrics, and extension ideas.

## Recommended use

1. Read `solutions/ch13_solutions.md` for the complete answer key.
2. Open the notebooks in numerical order.
3. Use the included environment and tests before assigning implementation work.
4. Treat the numerical results as pedagogical reference values, not benchmark claims.

## Repository structure

```text
ch13_instructor_guide/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/ch13_guide/
│   ├── __init__.py
│   ├── models.py
│   ├── utils.py
│   └── envs/reaching_env.py
├── solutions/ch13_solutions.md
├── notebooks/
│   ├── 00_setup_and_validation.ipynb
│   ├── 01_conceptual_and_mathematical_solutions.ipynb
│   ├── 02_robot_learning_environment.ipynb
│   ├── 03_behavior_cloning_pick_place.ipynb
│   ├── 04_sim_to_real_perturbation_suite.ipynb
│   ├── 05_offline_rl_dataset_design.ipynb
│   ├── 06_vla_benchmark_design.ipynb
│   └── 07_method_comparison.ipynb
└── tests/test_reaching_env.py
```

## Local installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
jupyter lab
```

## Google Colab

At the top of each notebook, run:

```python
!pip install -q "gymnasium>=1.0,<2" "torch>=2.5,<3" "numpy>=1.26,<3" \
    "matplotlib>=3.8,<4" "pandas>=2.2,<3" "scikit-learn>=1.4,<2"
```

Then upload the repository or clone it from GitHub and install it:

```python
!pip install -q -e .
```

## Reproducibility policy

All examples use explicit random seeds. Reports should include:

- environment version and seed;
- observation and action definitions;
- controller frequency and action scaling;
- train/validation/test split policy;
- termination and truncation rules;
- success metric and failure taxonomy;
- perturbation ranges;
- raw and safety-filtered actions;
- inference latency and hardware details.

## Software-interface note

The environment follows the Gymnasium API:
`reset()` returns `(observation, info)`, and `step()` returns
`(observation, reward, terminated, truncated, info)`.
The distinction matters because a time-limit truncation is not equivalent to a task-terminal transition.

## License

This instructor guide is intended to accompany the textbook project. Add the final project license before public release.