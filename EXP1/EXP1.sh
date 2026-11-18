#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${EXP1_PYTHON:-python3}

# Forward trainer options and plot this specific run when training completes.
exec "$python_bin" "$script_dir/EXP1_Run.py" --plot "$@"
