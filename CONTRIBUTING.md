# Contributing to DRPP Route Planner

Thank you for your interest in contributing to the DRPP Route Planner! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)

## Code of Conduct

This project follows a professional code of conduct. Be respectful, constructive, and collaborative.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic knowledge of graph algorithms (for core contributions)
- Familiarity with PyQt6 (for GUI contributions)

### First-Time Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Route_planner.git
   cd Route_planner
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Run Tests to Verify Setup**
   ```bash
   pytest
   ```

## Development Setup

### IDE Configuration

**VS Code** (recommended `.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true
}
```

**PyCharm**:
- Enable Black formatter in Settings → Tools → Black
- Set line length to 100
- Enable pytest as test runner

### Running the Application

```bash
# Core library (no GUI)
python -c "from drpp_core import parallel_cluster_routing; print('OK')"

# GUI application
python Route_Planner.py

# Pipeline script
python run_drpp_pipeline.py
```

## Making Changes

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

Example: `feature/add-vincenty-distance`

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): subject

body (optional)

footer (optional)
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(routing): add Vincenty distance calculation for improved accuracy

fix(gui): prevent crash when loading invalid KML files

docs(readme): update installation instructions for macOS

test(clustering): add parametrized tests for grid clustering
```

### Code Style Guidelines

1. **Follow PEP 8** with modifications:
   - Line length: 100 characters (not 79)
   - Use double quotes for strings
   - Use trailing commas in multi-line data structures

2. **Type Hints** are required for all new code:
   ```python
   def calculate_distance(coord1: Coordinate, coord2: Coordinate) -> float:
       """Calculate distance between coordinates."""
       ...
   ```

3. **Docstrings** are required for all public functions/classes:
   ```python
   def greedy_route(segments: List[Segment]) -> PathResult:
       """
       Route through segments using greedy nearest-neighbor.

       Args:
           segments: List of required segments to visit

       Returns:
           PathResult containing the optimized route

       Raises:
           RoutingError: If no valid route exists
       """
   ```

4. **Avoid**:
   - Global variables (except constants)
   - Circular imports
   - Magic numbers (use named constants)
   - Overly complex functions (>50 lines)

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=drpp_core

# Run specific test file
pytest tests/test_greedy_router.py

# Run tests matching pattern
pytest -k "test_routing"

# Run in parallel (faster)
pytest -n auto
```

### Writing Tests

1. **Location**: Place tests in `tests/` directory
2. **Naming**: Prefix test files with `test_`
3. **Structure**: Use pytest fixtures for reusable test data

**Example**:
```python
import pytest
from drpp_core import greedy_route_cluster

@pytest.fixture
def simple_graph():
    """Fixture providing a basic graph for testing."""
    # ... create test graph
    return graph

def test_greedy_routing_basic(simple_graph):
    """Test greedy routing on basic graph."""
    result = greedy_route_cluster(simple_graph, ...)
    assert result.distance > 0
    assert len(result.path) > 0

@pytest.mark.parametrize("num_segments,expected_time", [
    (10, 1.0),
    (100, 10.0),
])
def test_routing_performance(simple_graph, num_segments, expected_time):
    """Test routing performance scales linearly."""
    # ... test implementation
```

### Test Coverage Requirements

- **Core modules** (`drpp_core/`): Minimum 80% coverage
- **New features**: Must include tests
- **Bug fixes**: Add regression test

## Code Quality

### Automated Checks

Pre-commit hooks automatically run:
- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting
- **mypy**: Type checking
- **Bandit**: Security scanning

### Manual Code Review Checklist

Before submitting:
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Coverage meets requirements
- [ ] Documentation is updated
- [ ] Type hints are complete
- [ ] No security vulnerabilities
- [ ] Performance impact is acceptable

### Running Quality Checks Manually

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Type check
mypy drpp_core/

# Security scan
bandit -r drpp_core/

# All at once
pre-commit run --all-files
```

## Submitting Changes

### Pull Request Process

1. **Update Your Branch**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run All Checks**
   ```bash
   pytest
   pre-commit run --all-files
   ```

3. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature
   ```

4. **Create Pull Request**
   - Use descriptive title
   - Reference related issues
   - Provide detailed description
   - Add screenshots (for UI changes)
   - Check all items in PR template

### PR Title Format

```
[Type] Brief description (#issue-number)
```

Examples:
- `[Feature] Add Vincenty distance calculation (#123)`
- `[Fix] Prevent crash on invalid KML (#456)`
- `[Docs] Update installation guide (#789)`

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2

## Testing
How was this tested?

## Screenshots (if applicable)
Before/after screenshots

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Breaking changes documented
```

### Review Process

1. **Automated Checks**: GitHub Actions CI must pass
2. **Code Review**: At least one approval required
3. **Testing**: Manual testing for UI/UX changes
4. **Merge**: Squash and merge (maintainers only)

## Project Structure

```
Route_planner/
├── drpp_core/              # Production V4 core library
│   ├── clustering.py       # Geographic clustering
│   ├── distance_matrix.py  # Distance computations
│   ├── exceptions.py       # Custom exceptions
│   ├── geo.py             # Geographic utilities
│   ├── greedy_router.py   # Greedy routing algorithm
│   ├── parallel_executor.py # Parallel processing
│   ├── path_reconstruction.py
│   └── types.py           # Type definitions
├── legacy/                # Older implementations (read-only)
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
├── Route_Planner.py       # PyQt6 GUI application
├── drpp_pipeline.py       # Pipeline orchestrator
├── drpp_visualization.py  # Visualization generation
└── osm_speed_integration.py # OSM API integration
```

### Module Responsibilities

- **drpp_core**: Core DRPP solving logic (production-ready)
- **legacy**: Historical implementations (compatibility only)
- **Route_Planner.py**: GUI application (needs refactoring)
- **drpp_pipeline.py**: High-level pipeline orchestration
- **drpp_visualization.py**: Map and route visualization

## Areas Needing Contribution

### High Priority

1. **Route_Planner.py Refactoring**
   - Split 2,500-line file into MVC architecture
   - Extract business logic from GUI
   - Improve error handling

2. **Test Coverage**
   - Increase coverage to 80%+
   - Add integration tests
   - Add performance benchmarks

3. **Documentation**
   - API reference (Sphinx)
   - Architecture diagrams
   - Tutorial notebooks

### Medium Priority

4. **Performance Optimization**
   - Better spatial indexing (R-tree)
   - Async I/O for OSM API
   - Caching improvements

5. **New Features**
   - Additional routing algorithms
   - Custom cost functions
   - Real-time traffic integration

### Low Priority

6. **UI/UX Improvements**
   - Dark/light theme toggle
   - Keyboard shortcuts
   - Accessibility features

## Questions?

- **Issues**: [GitHub Issues](https://github.com/DillyVee/Route_planner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DillyVee/Route_planner/discussions)
- **Email**: See repository profile

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
