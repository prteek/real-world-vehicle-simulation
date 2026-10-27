#!/usr/bin/env bash
# Publish to Hugging Face (locally: `hf auth login` as prteek; in CI: HF_TOKEN):
#   dataset       prteek/drive-cycle: data/data.csv, road_load.json (the frozen road-load model from
#                 vehicle_sim.drive_cycle) and dataset/README.md as the card; by hand, since the data is not in git
#                 (CI publishes only the card)
#   dataset-card  only the dataset card
#   space         prteek/electric-truck-battery-sizing: space/ (app, card, requirements) plus vehicle_sim; the app
#                 downloads the data from the dataset
# Usage: ./publish.sh [message] [dataset|dataset-card|space ...]   (default: dataset space)
# Set HF_CLI=hf to use an installed hf CLI instead of `uvx hf`.
set -euo pipefail
cd "$(dirname "$0")"
msg="${1:-Update}"
shift || true
if [ $# -gt 0 ]; then targets=("$@"); else targets=(dataset space); fi
read -r -a hf <<< "${HF_CLI:-uvx --from huggingface_hub hf}"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

for t in "${targets[@]}"; do
  case "$t" in
    dataset)
      mkdir "$stage/dataset"
      cp data/data.csv "$stage/dataset/"
      ${PYTHON:-uv run python} -c "import json; from vehicle_sim.drive_cycle import road_load; print(json.dumps(road_load(), indent=1))" \
        > "$stage/dataset/road_load.json"
      cp dataset/README.md "$stage/dataset/README.md"
      "${hf[@]}" upload prteek/drive-cycle "$stage/dataset" . --repo-type dataset --commit-message "$msg" ;;
    dataset-card)
      "${hf[@]}" upload prteek/drive-cycle dataset/README.md README.md --repo-type dataset --commit-message "$msg" ;;
    space)
      mkdir -p "$stage/space/vehicle_sim"
      cp space/app.py space/README.md space/requirements.txt "$stage/space/"
      cp vehicle_sim/__init__.py vehicle_sim/drive_cycle.py vehicle_sim/simulation.py "$stage/space/vehicle_sim/"
      "${hf[@]}" upload prteek/electric-truck-battery-sizing "$stage/space" . --repo-type space --commit-message "$msg" ;;
    *)
      echo "unknown target: $t" >&2; exit 1 ;;
  esac
done
