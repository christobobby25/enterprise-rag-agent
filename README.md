## Development Workflow

This project uses GitHub Actions for Continuous Integration.

Every pull request automatically runs:
- Python dependency installation
- Ruff linting
- Pytest unit tests

Changes must pass CI checks before merging into main.