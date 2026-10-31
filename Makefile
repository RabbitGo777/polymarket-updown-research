.PHONY: notebooks test ticks

notebooks:
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 *.ipynb

test:
	ruff check .
	pytest -q

ticks:
	python scripts/build_tick_lag.py
