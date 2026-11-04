import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from voice_ai.evaluate_student import parse_args, run_single_evaluation


if __name__ == "__main__":
    run_single_evaluation(parse_args())
