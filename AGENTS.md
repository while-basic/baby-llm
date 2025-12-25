# Repository Agent Guide

This repository contains several sibling experimental projects. Unless otherwise specified, work from the repository root and be explicit about which project you are modifying.

## Development expectations
- Match the existing style of the files you touch; avoid introducing new tooling unless necessary.
- When you change code or tests inside `NeuralChild-main/`, run `python run_tests.py` from that directory (or a more targeted equivalent) before opening a PR.
- Keep documentation updates alongside functional changes when the behavior or commands change.

## Pull request messaging
- Summaries should list the key functional changes and mention the project path you touched (e.g., `NeuralChild-main`).
- Always include the commands you ran for verification under a distinct testing section.
