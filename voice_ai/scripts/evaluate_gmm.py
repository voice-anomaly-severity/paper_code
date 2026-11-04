import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from voice_ai.evaluate_gmm import main, parse_args


if __name__ == "__main__":
    main(parse_args())
