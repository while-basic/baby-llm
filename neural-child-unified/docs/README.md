# Documentation

This directory contains the Sphinx documentation for the Baby LLM - Unified Neural Child Development System.

## Building Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Build Documentation

Using Make (Linux/Mac):

```bash
cd docs
make html
```

Using Sphinx directly:

```bash
cd docs
sphinx-build -b html . _build/html
```

### View Documentation

After building, open `_build/html/index.html` in your browser.

## Read the Docs

The documentation is configured for Read the Docs. The configuration is in `.readthedocs.yaml` at the project root.

### Setup on Read the Docs

1. Go to [Read the Docs](https://readthedocs.org/)
2. Import your repository
3. The documentation will build automatically using the configuration in `.readthedocs.yaml`

## Documentation Structure

- `index.rst` - Main documentation entry point
- `getting-started.md` - Quick start guide
- `installation.md` - Installation instructions
- `usage.rst` - Usage documentation
- `api-reference.rst` - API documentation
- `modules.md` - Module documentation
- `architecture.rst` - Architecture overview
- `api/` - Detailed API documentation by category

## Adding Documentation

1. Add new `.rst` or `.md` files to this directory
2. Include them in `index.rst` using the `toctree` directive
3. For markdown files, use the `myst_parser` extension (already configured)
