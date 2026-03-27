# Security

## Supported versions

Security fixes are applied to the **latest minor release** in the current **0.x** line unless otherwise stated in a security advisory.

## Reporting a vulnerability

Please report security issues **privately** to the maintainers (use the contact method listed in the repository README or organization profile once published). Do not open a public issue for undisclosed vulnerabilities.

Include:

- Affected versions or commit range
- Steps to reproduce and impact assessment
- Suggested fix (if any)

We aim to acknowledge reports within **72 hours** and coordinate disclosure and release timelines with the reporter.

## Scope

- **In scope:** EdgeAdapt Python/Rust code, bundled examples, and documented CLI entry points.
- **Out of scope:** upstream vulnerabilities in ONNX Runtime, PyTorch, OS kernels, or third-party models unless EdgeAdapt is the direct cause.

## Process

1. Triage and reproduce.
2. Develop and test a fix on a private branch.
3. Prepare a release and release notes.
4. Coordinate public disclosure after users can upgrade.

This document is intentionally lightweight for early-stage development; it will evolve with maintainer capacity and adoption.
