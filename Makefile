.PHONY: dev-install lint lint-rust lint-py test test-rust bench docs

dev-install:
	maturin develop --extras dev

lint: lint-py lint-rust

lint-py:
	ruff check python tests/python
	ruff format --check python tests/python

lint-rust:
	cargo fmt --manifest-path rust/edgeadapt_core/Cargo.toml --check
	cargo clippy --manifest-path rust/edgeadapt_core/Cargo.toml -- -D warnings

test:
	pytest tests/python -q

test-rust:
	cargo test --manifest-path rust/edgeadapt_core/Cargo.toml

bench:
	pytest tests/python -q -m bench || true

docs:
	mkdocs build -f docs/mkdocs.yml
