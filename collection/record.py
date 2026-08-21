"""Record trajectories with a selectable rollout context length.

The model's trained window is fixed in its bundle. `--context-length`
controls how much rollout history the inference KV cache retains.

Example:
    python collection/record.py --env-id Humanoid-v5 --out raw/humanoid \
        --episodes 100 --context-length 128
"""

import sys
from pathlib import Path

# Direct execution puts `collection/` rather than the repository root on
# sys.path. Add the checkout root so the shared recorder can be imported.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.deploy.record import main


if __name__ == "__main__":
    main()
