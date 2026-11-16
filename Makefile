.PHONY: test lint bench sweep serve
test:
	pytest -m "not slow"
lint:
	ruff check src tests bench && mypy src
bench:
	python bench/humaneval.py --n 12 --max-new 200 --k 4
sweep:
	python bench/sweep.py
serve:
	python -m specinfer.server.app
