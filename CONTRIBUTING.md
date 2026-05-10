# Contributing to BSDPose

Thank you for your interest in contributing! 🏸

## How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m 'Add some feature'`
4. **Push** to the branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

## Development Setup

```bash
git clone https://github.com/lechan775/BSDPose.git
cd BSDPose
pip install -r requirements.txt
```

## Code Style

- Python 3.10+ with type annotations (`from __future__ import annotations`)
- PEP 8 formatting
- Docstrings for public functions
- Use `pathlib.Path` for file paths

## Adding New Experiments

1. Add experiment definition in `configs/config.yaml` under `experiments:`
2. Implement model variant in appropriate module
3. Run `python eval/eval_ablation.py` to verify

## Issues

Use [GitHub Issues](https://github.com/lechan775/BSDPose/issues) for bug reports and feature requests.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
