# Chapter 1 Instructor Guide and Lab Materials

This package contains instructor-facing solutions and runnable lab code for Chapter 1 exercises from *Deep Reinforcement Learning for Autonomous Intelligent Systems: Theory, Implementation, and Applications*.

## Contents

```text
ch1_instructor_package/
├── README.md
├── requirements.txt
├── chapter1_instructor_solutions.md
├── src/
│   ├── gridworld.py
│   ├── algorithms.py
│   └── run_experiments.py
└── notebooks/
    └── chapter1_grid_rl_colab.ipynb
```

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:

- `numpy`
- `matplotlib`

No Gymnasium dependency is required. The environment follows the same API idea used by modern RL libraries by returning:

```python
next_state, reward, terminated, truncated, info
```

## Quick Start

From this directory:

```bash
python src/run_experiments.py
```

This runs TD(0) policy evaluation, SARSA control, and Q-learning control.

## Instructor Use

The Markdown file `chapter1_instructor_solutions.md` provides conceptual and mathematical solutions for all exercises. The code files support Exercises 13--15 and can be used as a companion GitHub lab.
