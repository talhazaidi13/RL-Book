# Instructor Checklist

Before release:

- [ ] Run `pytest -q`.
- [ ] Execute all notebooks from a clean runtime.
- [ ] Replace proxy tasks with the course simulator when appropriate.
- [ ] Confirm that no notebook contains private credentials or machine-specific paths.
- [ ] Freeze a tested dependency lock file for the semester.
- [ ] Record the tested Python, Gymnasium, and PyTorch versions.
- [ ] Add the project license and citation information.
- [ ] Store large datasets outside Git and provide download scripts.
- [ ] Require students to report terminated and truncated transitions separately.
- [ ] Require closed-loop metrics, failure labels, and reproducibility metadata.