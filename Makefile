.PHONY: help refresh refresh-force test

## help: list available targets
help:
	@grep -E '^##' Makefile | sed 's/## /  /'

## refresh: re-pull the daily-updated GitHub dataset (6h cache) and rebuild predictions.json
refresh:
	cd model && uv run python -m fifa_pred.build_predictions

## refresh-force: same as refresh but bypass the dataset cache (force re-download)
refresh-force:
	cd model && FIFA_PRED_FORCE_FETCH=1 uv run python -m fifa_pred.build_predictions

## test: run the model test suite
test:
	cd model && uv run pytest
