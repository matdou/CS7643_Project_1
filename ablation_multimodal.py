import argparse
from copy import deepcopy

from evaluate_multimodal import evaluate, parse_args as eval_parse_args


def run_ablation(base_args: argparse.Namespace) -> None:
    scenarios = {
        "all": {"drop_video": False, "drop_audio": False, "drop_prompt": False},
        "video_only": {"drop_video": False, "drop_audio": True, "drop_prompt": True},
        "audio_only": {"drop_video": True, "drop_audio": False, "drop_prompt": True},
        "prompt_only": {"drop_video": True, "drop_audio": True, "drop_prompt": False},
        "no_video": {"drop_video": True, "drop_audio": False, "drop_prompt": False},
        "no_audio": {"drop_video": False, "drop_audio": True, "drop_prompt": False},
        "no_prompt": {"drop_video": False, "drop_audio": False, "drop_prompt": True},
    }

    for name, drops in scenarios.items():
        args = deepcopy(base_args)
        args.drop_video = drops["drop_video"]
        args.drop_audio = drops["drop_audio"]
        args.drop_prompt = drops["drop_prompt"]
        metrics = evaluate(args)
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    args = eval_parse_args()
    run_ablation(args)
