"""Public-only goal/ball grounding, automatic roles, and short bidirectional tracking.

Inference functions receive pixels only. Public clicks are used by the separate audit.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ball_goal_pilot"
DATA = Path(json.loads((ROOT / "dataset_path.json").read_text())["path"])
os.environ.setdefault("TORCH_HOME", str(ROOT / "models/rtmlib"))
os.environ.setdefault("HF_HOME", str(ROOT / "models/.hf-cache"))
CONFIG = {
    "model": "IDEA-Research/grounding-dino-tiny",
    "prompt": "a soccer goal. a soccer ball.",
    "box_threshold": 0.20, "text_threshold": 0.20,
    "anchor_offsets_s": [-0.35, -0.16, 0.0],
    "person_model": "rtmlib Wholebody balanced",
    "version": 1,
}
ROLE_CONFIG = {"version": 1, "max_foot_ball_distance_heights": 0.8,
               "goal_min_score": 0.30, "ball_min_score": 0.25,
               "tracking": "OpenCV CSRT, independent forward/backward from seed"}


def prepare_sample():
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(ROOT / "outputs/perception_pilot/sample.csv")
    dev = sample[sample.audit_split == "development"].head(12)
    held = sample[sample.audit_split == "holdout"]
    assert set(dev.source_id).isdisjoint(held.source_id)
    subset = pd.concat([dev, held], ignore_index=True)
    path = OUT / "sample.csv"
    if path.exists():
        assert pd.read_csv(path)["clip"].tolist() == subset["clip"].tolist()
    else:
        subset.to_csv(path, index=False)
    return subset


def read_tail(path, seconds=0.60):
    """Sequential decode preserves actual presentation timestamps, including VFR."""
    cap = cv2.VideoCapture(str(path))
    tail = deque()
    index, previous = 0, -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if timestamp <= previous:
            raise RuntimeError(f"Non-increasing timestamps: {path}, frame {index}")
        previous = timestamp
        tail.append((index, timestamp, frame))
        while len(tail) > 1 and timestamp - tail[0][1] > seconds:
            tail.popleft()
        index += 1
    cap.release()
    if not tail:
        raise RuntimeError(f"Cannot decode {path}")
    return list(tail)


def clamp_box(box, width, height):
    b = np.asarray(box, dtype=float).copy()
    b[[0, 2]] = np.clip(b[[0, 2]], 0, width - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, height - 1)
    return b.tolist()


class Detector:
    def __init__(self):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from rtmlib import Wholebody
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        self.torch = torch
        model_path = ROOT / "models/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True).eval()
        self.people = Wholebody(mode="balanced", backend="onnxruntime", device="cpu",
                               to_openpose=False)

    def detect(self, frame):
        h, w = frame.shape[:2]
        inputs = self.processor(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                text=CONFIG["prompt"], return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=CONFIG["box_threshold"],
            text_threshold=CONFIG["text_threshold"], target_sizes=[(h, w)])[0]
        objects = []
        for b, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
            # Ambiguous joint phrases are excluded rather than silently assigned a class.
            kind = "goal" if "goal" in label and "ball" not in label else (
                "ball" if "ball" in label and "goal" not in label else "ambiguous")
            objects.append({"bbox": clamp_box(b.tolist(), w, h), "score": float(score),
                            "label": label, "kind": kind})
        persons = []
        for b in self.people.det_model(frame):
            box = clamp_box(b, w, h)
            person = {"bbox": box}
            if box[3] - box[1] >= 0.035 * h:
                kp, conf = self.people.pose_model(frame, bboxes=[box])
                person.update(keypoints=kp[0, :23].tolist(), scores=conf[0, :23].tolist())
            persons.append(person)
        return {"width": w, "height": h, "objects": objects, "persons": persons}


def detect_clips(sample, limit):
    detector = Detector()
    config_hash = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    started = time.time()
    for row in sample.head(limit).itertuples():
        dest = OUT / "detections" / (row.clip + ".json")
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            assert json.loads(dest.read_text())["config_hash"] == config_hash
            continue
        frames = read_tail(DATA / "train" / row.clip)
        end = frames[-1][1]
        anchors = sorted(set(min(range(len(frames)), key=lambda j: abs(
            frames[j][1] - (end + offset))) for offset in CONFIG["anchor_offsets_s"]))
        records = []
        for j in anchors:
            index, timestamp, frame = frames[j]
            result = detector.detect(frame)
            result.update(frame_index=index, timestamp=timestamp)
            records.append(result)
        dest.write_text(json.dumps({"clip": row.clip, "config_hash": config_hash,
                                    "config": CONFIG, "anchors": records}, indent=2) + "\n")
        print(f"Detected {row.clip}: " + ", ".join(
            f'{len(a["objects"])} objects/{len(a["persons"])} people' for a in records)
            + f"; total {time.time()-started:.1f}s", flush=True)


def feet(person):
    box = person["bbox"]
    if "keypoints" in person:
        kp, conf = np.asarray(person["keypoints"]), np.asarray(person["scores"])
        valid = conf[15:23] > 0.30
        if valid.any():
            return kp[15:23][valid]
    return np.asarray([[(box[0] + box[2]) / 2, box[3]]])


def role_candidates(anchor):
    """Rank scene-compatible pairs without positions, labels, or source metadata."""
    w, h = anchor["width"], anchor["height"]
    goals = [o for o in anchor["objects"] if o["kind"] == "goal"
             and o["score"] >= ROLE_CONFIG["goal_min_score"]
             and o["bbox"][2] - o["bbox"][0] > 0.08 * w]
    balls = [o for o in anchor["objects"] if o["kind"] == "ball"
             and o["score"] >= ROLE_CONFIG["ball_min_score"]
             and 1 < o["bbox"][2] - o["bbox"][0] < 0.06 * w
             and 0.4 < (o["bbox"][2] - o["bbox"][0]) /
             max(1, o["bbox"][3] - o["bbox"][1]) < 2.5]
    candidates = []
    for goal in goals:
        gx1, gy1, gx2, gy2 = goal["bbox"]
        gw, gh = gx2 - gx1, max(1, gy2 - gy1)
        keeper_options = []
        for gi, person in enumerate(anchor["persons"]):
            b = person["bbox"]
            px, py = np.median(feet(person), axis=0)
            height_ratio = (b[3] - b[1]) / gh
            if not (gx1 - 0.15 * gw <= px <= gx2 + 0.15 * gw
                    and gy1 + 0.30 * gh <= py <= gy2 + 0.40 * gh
                    and 0.20 <= height_ratio <= 1.60):
                continue
            # Goal box is only a coarse proxy for the goal line, not calibration.
            cost = abs(py - gy2) / gh + 0.20 * abs(px - (gx1 + gx2) / 2) / gw
            cost += 0.30 * abs(np.log(height_ratio / 0.65))
            keeper_options.append((cost, gi))
        for ball in balls:
            b = ball["bbox"]
            center = np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            if center[1] < gy1 + 0.3 * gh:
                continue
            for ki, person in enumerate(anchor["persons"]):
                pb = person["bbox"]
                ph = max(1, pb[3] - pb[1])
                if ph < 0.15 * gh:
                    continue
                distance_px = float(np.linalg.norm(feet(person) - center, axis=1).min())
                distance = distance_px / ph
                if distance > ROLE_CONFIG["max_foot_ball_distance_heights"]:
                    continue
                for keeper_cost, gi in keeper_options:
                    if ki == gi:
                        continue
                    # Reject nearby crowd pairs and goalkeeper-as-kicker assignments.
                    gp = anchor["persons"][gi]["bbox"]
                    separation = np.linalg.norm(np.asarray([(pb[0]+pb[2])/2, pb[3]]) -
                                                np.asarray([(gp[0]+gp[2])/2, gp[3]]))
                    if separation < 0.60 * max(ph, gp[3] - gp[1]):
                        continue
                    score = (2.0 * ball["score"] + goal["score"] - 1.3 * distance
                             - 0.7 * keeper_cost - 2 * distance_px / h)
                    candidates.append({"score": float(score), "kicker_index": ki,
                                       "keeper_index": gi, "goal": goal, "ball": ball,
                                       "foot_ball_distance_heights": distance,
                                       "keeper_geometry_cost": float(keeper_cost)})
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def track_person(frames, seed_index, box):
    """Short-window pilot, not full-clip identity tracking or occlusion recovery."""
    output = {frames[seed_index][0]: box}
    for step in [-1, 1]:
        tracker = cv2.TrackerCSRT_create()
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        tracker.init(frames[seed_index][2], (x1, y1, max(2, x2-x1), max(2, y2-y1)))
        stop = -1 if step == -1 else len(frames)
        for j in range(seed_index + step, stop, step):
            ok, rect = tracker.update(frames[j][2])
            if not ok:
                break
            x, y, width, height = rect
            h, w = frames[j][2].shape[:2]
            output[frames[j][0]] = clamp_box([x, y, x+width, y+height], w, h)
    return output


def assign_clip(detections, frames):
    candidates = []
    for i, a in enumerate(detections["anchors"]):
        for c in role_candidates(a):
            candidates.append(dict(c, anchor_index=i))
    if not candidates:
        return {"status": "abstain", "reason": "No compatible goal-ball-person pair",
                "tracks": {}}
    best = max(candidates, key=lambda c: c["score"])
    anchor = detections["anchors"][best["anchor_index"]]
    seed = next(j for j, f in enumerate(frames) if f[0] == anchor["frame_index"])
    tracks = {}
    for role in ["kicker", "keeper"]:
        box = anchor["persons"][best[role + "_index"]]["bbox"]
        tracks[role] = track_person(frames, seed, box)
    return {"status": "pair", "seed_frame": anchor["frame_index"],
            "seed_timestamp": anchor["timestamp"], "selection": best, "tracks": tracks}


def assign_clips(sample):
    for row in sample.itertuples():
        source = OUT / "detections" / (row.clip + ".json")
        if not source.exists():
            continue
        frames = read_tail(DATA / "train" / row.clip)
        result = assign_clip(json.loads(source.read_text()), frames)
        result.update(clip=row.clip, role_config=ROLE_CONFIG,
                      detection_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        dest = OUT / "predictions" / source.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Assigned {row.clip}: {result['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=38)
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--stage", choices=["detect", "assign"], default="detect")
    args = parser.parse_args()
    sample = prepare_sample()
    if args.split != "all":
        sample = sample[sample.audit_split == args.split]
    if args.stage == "detect":
        detect_clips(sample, args.limit)
    else:
        assign_clips(sample.head(args.limit))


if __name__ == "__main__":
    main()
