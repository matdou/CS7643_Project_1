#!/usr/bin/env python3
from pathlib import Path
import json
import torch
from query_encoder.clip_query_encoder import CLIPQueryEncoder

PROMPTS_TRAIN = Path(__file__).parent / "prompts_train.yaml"
PROMPTS_TEST  = Path(__file__).parent / "prompts_test.yaml"
EMB_DIR = Path(__file__).parent / "embeddings"
CLASS_MAT_PT = EMB_DIR / "class_matrix.pt"
CLASS_NAMES_JSON = EMB_DIR / "class_names.json"

def load_or_build_prototypes(enc: CLIPQueryEncoder):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    if CLASS_MAT_PT.exists() and CLASS_NAMES_JSON.exists():
        class_mat = torch.load(CLASS_MAT_PT)
        class_names = json.loads(CLASS_NAMES_JSON.read_text())
        return class_mat, class_names
    class_mat, class_names = enc.build_class_matrix(str(PROMPTS_TRAIN))
    torch.save(class_mat.cpu(), CLASS_MAT_PT)
    CLASS_NAMES_JSON.write_text(json.dumps(class_names, indent=2))
    return class_mat, class_names

def check_shape_and_norm(enc: CLIPQueryEncoder):
    v = enc.encode_query(["referee shows a red card"])[0]
    assert v.shape[-1] == 512, f"Expected 512-dim, got {v.shape[-1]}"
    n = float(v.norm())
    assert 0.99 < n < 1.01, f"Expected unit norm, got {n:.4f}"

def zero_shot_sanity(enc: CLIPQueryEncoder, class_mat, class_names):
    tests = [
        ("the ball goes into the net", "Goal"),
        ("referee points to the spot", "Penalty"),
        ("corner kick taken from the flag", "Corner"),
        ("assistant raises the flag for offside", "Offside"),
        ("referee shows a red card", "Red card"),
    ]
    ok = 0
    for s, expected in tests:
        top = enc.classify(s, class_mat, class_names, topk=3)
        preds = [p[0] for p in top]
        hit = expected in preds
        print(f"{s!r} -> {preds} | expected: {expected} | {'OK' if hit else 'MISS'}")
        ok += int(hit)
    assert ok >= 4, f"Zero-shot sanity too low ({ok}/5). Check prompts.yaml or CLIP install."


def evaluate_on_test(enc: CLIPQueryEncoder, class_mat, class_names, prompts_test):
    import yaml
    test_data = yaml.safe_load(Path(prompts_test).read_text())

    total, correct_top1, correct_top3 = 0, 0, 0
    for action, sentences in test_data.items():
        for s in sentences:
            preds = [p[0] for p in enc.classify(s, class_mat, class_names, topk=3)]
            total += 1
            if preds[0] == action:
                correct_top1 += 1
            if action in preds:
                correct_top3 += 1
    print(f"\nZero-shot CLIP performance on test set:")
    print(f"Top-1 accuracy: {correct_top1/total:.3f}")
    print(f"Top-3 accuracy: {correct_top3/total:.3f}")

def cosine_consistency(enc: CLIPQueryEncoder):
    a1 = enc.encode_query(["player scores a goal"])[0]
    a2 = enc.encode_query(["player scores a goal"])[0]
    b  = enc.encode_query(["a throw-in is taken"])[0]
    same = torch.cosine_similarity(a1, a2, dim=0).item()
    diff = torch.cosine_similarity(a1, b, dim=0).item()
    print(f"cos(same)={same:.4f}  cos(diff)={diff:.4f}")
    assert same > 0.999, "Same sentence should be ~identical"
    assert diff < 0.7, "Different actions should be less similar"

def main():
    enc = CLIPQueryEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
    class_mat, class_names = load_or_build_prototypes(enc)
    zero_shot_sanity(enc, class_mat, class_names)
    #cosine_consistency(enc)

    evaluate_on_test(enc, class_mat, class_names, PROMPTS_TEST)

    print("\n[PASS] CLIP query encoder basic tests passed.")

if __name__ == "__main__":
    main()
