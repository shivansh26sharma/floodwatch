#!/usr/bin/env bash
# Run the whole pipeline end to end.
#   ./scripts/run_pipeline.sh            # horizon 1 day
#   ./scripts/run_pipeline.sh 3          # horizon 3 days
set -euo pipefail

HORIZON="${1:-1}"
cd "$(dirname "$0")/.."

echo "=== 1/5 fixture ==="
python -m src.data.make_fixture

echo "=== 2/5 dataset ==="
python -m src.data.build_dataset

echo "=== 3/5 features (horizon ${HORIZON}d) ==="
python -m src.features.build_features --horizon "${HORIZON}"

echo "=== 4/5 train + evaluate ==="
python -m src.models.evaluate --horizon "${HORIZON}"

echo "=== 5/6 explain ==="
python -m src.models.explain --horizon "${HORIZON}" --model random_forest

echo "=== 6/6 forecasting ==="
# Add --no-lstm to skip the neural net on a slow machine.
python -m src.models.forecast

echo
echo "Done. Launch the dashboard with:"
echo "    streamlit run dashboard/app.py"
