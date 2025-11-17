## Description

<!-- Provide a brief summary of the changes in this PR -->

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Dependencies update
- [ ] CI/CD changes

## Related Issues

<!-- Link to related issues. Use "Fixes #123" or "Closes #123" to auto-close issues when merged -->

Fixes #
Related to #

## Changes Made

<!-- Provide a detailed list of changes -->

-
-
-

## Testing

<!-- Describe the tests you ran and how to reproduce -->

### Test Environment

- Python Version:
- OS:
- Dataset Size (if applicable):

### Test Cases

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed
- [ ] Tested with sample KML files
- [ ] Performance benchmarking done (for perf changes)

### Test Commands

```bash
# Commands used to test this PR
pytest tests/ -v
python run_drpp_pipeline.py test_data.kml v4
```

## Performance Impact

<!-- For performance-related changes or large features -->

- [ ] No performance impact
- [ ] Performance improved (provide benchmarks)
- [ ] Performance degraded (justified because...)
- [ ] Not applicable

**Benchmarks** (if applicable):
```
Before:
After:
```

## Breaking Changes

<!-- List any breaking changes and migration steps -->

- [ ] No breaking changes
- [ ] Breaking changes (detailed below)

**Migration Guide** (if applicable):
```python
# Old way
...

# New way
...
```

## Documentation

- [ ] Code comments added/updated
- [ ] Docstrings added/updated
- [ ] README.md updated (if needed)
- [ ] CHANGELOG.md updated
- [ ] Documentation in `docs/` updated (if needed)
- [ ] Type hints added for new code

## Code Quality

- [ ] Code follows project style guidelines (Black, isort, flake8)
- [ ] Pre-commit hooks pass
- [ ] No new linting warnings
- [ ] Type checking passes (mypy)
- [ ] All tests pass locally

## Security

<!-- For security-sensitive changes -->

- [ ] No security impact
- [ ] Security review needed
- [ ] Dependency vulnerabilities checked (safety)
- [ ] Input validation added/reviewed

## Screenshots/Examples

<!-- If applicable, add screenshots or code examples -->

```python
# Example usage of new feature
from drpp_core import new_function

result = new_function(param1, param2)
```

## Checklist

- [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines
- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Additional Notes

<!-- Any additional information, context, or screenshots -->

---

**Reviewer Notes**

<!-- For reviewers to add comments during review -->
