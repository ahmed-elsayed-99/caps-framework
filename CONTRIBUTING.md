# Contributing to CAPS Framework

## Development Setup

```bash
git clone https://github.com/ahmed-elsayed-99/caps-framework.git
cd caps-framework
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Code Standards

All code must pass the following before submission:

```bash
black caps/ tests/
isort caps/ tests/
flake8 caps/ tests/ --max-line-length=100 --ignore=E203,W503
mypy caps/ --ignore-missing-imports
pytest tests/ --cov=caps --cov-report=term-missing
```

## Branch Strategy

- `main` — stable releases only
- `develop` — integration branch for features
- `feature/your-feature-name` — individual feature branches

## Pull Request Requirements

- All tests must pass with coverage above 80%.
- New modules must include corresponding test files in `tests/`.
- New public functions must include docstrings with parameter types and return types.
- Commits follow conventional commit format: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.

## Adding a New Signal Dimension

To extend the signal matrix beyond the five base dimensions:

1. Add the column to `REQUIRED_COLUMNS` or `OPTIONAL_COLUMNS` in `caps/acquisition/signal_matrix.py`.
2. Document the column in `data/README.md`.
3. Update `SIGNAL_LABELS` in `caps/segmentation/profiles.py` and `caps/visualization/signal_plots.py`.
4. Add a test case in `tests/test_acquisition.py`.

## Reporting Issues

Open an issue at `https://github.com/ahmed-elsayed-99/caps-framework/issues` with:
- Python version
- CAPS version (`caps.__version__`)
- Minimal reproducible example
- Full traceback