# Contributing to HAC Diabetes Segmentation

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/ghada-59/HAC_diabetes_Segmentation.git
cd HAC_diabetes_Segmentation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install pytest pytest-cov
```

## Code Standards

- **PEP 8** compliance
- Type hints for all functions
- Docstrings (Google style)
- Tests for all new code

## Testing

```bash
pytest -v
pytest --cov=validation_metrics
```

## Pull Request Process

1. Create feature branch
2. Add tests
3. Run linting
4. Submit PR

## License

MIT License - see LICENSE file