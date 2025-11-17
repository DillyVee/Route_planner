# Contributing to Route Planner DRPP

Thank you for your interest in contributing to Route Planner! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, package versions)
- **Sample KML files** if applicable (anonymized if needed)
- **Error messages and stack traces**

Use the bug report template when creating issues.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear use case** - Why is this enhancement needed?
- **Detailed description** - How should it work?
- **Alternatives considered** - What other approaches did you think about?
- **Performance implications** - Will it affect large datasets?

Use the feature request template when creating issues.

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Set up development environment** (see below)
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run the test suite** and ensure all tests pass
7. **Commit your changes** with clear, descriptive messages
8. **Push to your fork** and submit a pull request

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Route_planner.git
cd Route_planner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements_production.txt
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=drpp_core --cov-report=html

# Run specific test file
python -m pytest tests/test_clustering.py -v

# View coverage report
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

### Code Quality Checks

```bash
# Format code with Black
black drpp_core/ drpp_pipeline.py drpp_visualization.py

# Sort imports with isort
isort drpp_core/ drpp_pipeline.py drpp_visualization.py

# Check with flake8
flake8 drpp_core/ --max-line-length=100

# Type check with mypy
mypy drpp_core/ --ignore-missing-imports
```

Pre-commit hooks will automatically run these checks before each commit.

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) with line length of 100 characters
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints for all function signatures
- Write docstrings for all public functions, classes, and modules

### Docstring Format

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """Brief description of what the function does.

    More detailed explanation if needed. Can span multiple paragraphs.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided

    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output
    """
```

### Testing Guidelines

- Write tests for all new functionality
- Aim for at least 70% code coverage
- Use descriptive test names: `test_greedy_router_handles_large_clusters`
- Include edge cases and error conditions
- Mock external dependencies (file I/O, network calls)

### Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(greedy_router): add on-demand Dijkstra mode for large datasets

Implement on-demand shortest path computation to avoid all-pairs
distance matrix for clusters with >1000 endpoints. Results in
10-100x performance improvement.

Closes #42
```

```
fix(kml_parser): handle malformed XML with fallback sanitization

Add regex-based XML sanitization before parsing to handle common
encoding issues in MapPlus KML files.

Fixes #38
```

## Project Structure

```
Route_planner/
├── drpp_core/              # V4 production core (main package)
│   ├── __init__.py         # Public API exports
│   ├── types.py            # Type definitions
│   ├── greedy_router.py    # Greedy routing algorithm
│   ├── distance_matrix.py  # Distance computation
│   ├── clustering.py       # Geographic clustering
│   └── ...
├── drpp_pipeline.py        # High-level pipeline orchestrator
├── drpp_visualization.py   # Visualization generation
├── tests/                  # Unit tests (mirror drpp_core structure)
├── docs/                   # Documentation
├── legacy/                 # Historical implementations (do not modify)
└── ...
```

### Where to Add Code

- **New routing algorithms** → `drpp_core/` (add to `__init__.py` exports)
- **Visualization features** → `drpp_visualization.py`
- **Pipeline enhancements** → `drpp_pipeline.py`
- **Utilities** → Appropriate module in `drpp_core/`
- **Tests** → `tests/test_<module>.py`

## Performance Considerations

When contributing code that processes large datasets:

1. **Profile first** - Use `drpp_core.profiling` to measure performance
2. **Consider memory** - Large distance matrices can exceed RAM
3. **Test with realistic data** - Try with 1000+ segments
4. **Document complexity** - Add Big-O notation in docstrings
5. **Provide alternatives** - On-demand vs precomputed modes

## Documentation

Update documentation when:

- Adding new features → Update `docs/PIPELINE_GUIDE.md` and docstrings
- Changing APIs → Update `docs/PRODUCTION_REFACTOR_GUIDE.md`
- Fixing bugs → Update `CHANGELOG.md`
- Adding dependencies → Update `README.md` and `requirements*.txt`

## Release Process

Maintainers will handle releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create GitHub release with tag `vX.Y.Z`
4. GitHub Actions will automatically build and publish to PyPI

## Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: See `example_production_usage.py`
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- `CHANGELOG.md` for feature additions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Route Planner! 🎉
