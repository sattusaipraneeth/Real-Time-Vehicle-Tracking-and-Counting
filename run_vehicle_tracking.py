import argparse
import os
import yaml
from collections import defaultdict
import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--weights", type=str, default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--line-ratio", type=float, default=0.62)
    p.add_argument("--output", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    src = args.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        out_path = args.output or "vehicle_tracking_output.mp4"
    else:
        cap = cv2.VideoCapture(src)
        base, ext = os.path.splitext(os.path.basename(src))
        out_path = args.output or f"{base}_tracked.mp4"
    if not cap.isOpened():
        print("Error: cannot open video source")
        raise SystemExit(1)
    ok, frame0 = cap.read()
    if not ok:
        print("Error: cannot read first frame")
        raise SystemExit(1)
    h, w = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    line_y = int(h * args.line_ratio)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    tracker_yaml = os.path.join(os.path.dirname(__file__), "bytetrack_low.yaml")
    cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.15,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.12,
        "track_buffer": 90,
        "match_thresh": 0.8,
        "fuse_score": True,
    }
    with open(tracker_yaml, "w") as f:
        yaml.safe_dump(cfg, f)
    model = YOLO(args.weights)
    allowed = {"bicycle", "car", "motorcycle", "bus", "truck"}
    prev_y = {}
    counted_ids = set()
    counts = defaultdict(int)
    font = cv2.FONT_HERSHEY_SIMPLEX
    id_scale, id_thick = 0.55, 2
    count_scale, count_thick = 0.90, 2
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(
            frame,
            persist=True,
            tracker=tracker_yaml,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 3)
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            tids = res.boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, tid in zip(boxes, clss, tids):
                name = model.names.get(int(c), str(c))
                if name not in allowed:
                    continue
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                label = f"ID:{tid} {name}"
                (tw, th), _ = cv2.getTextSize(label, font, id_scale, id_thick)
                y_text = max(20, y1 - 8)
                cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + 3, y_text), font, id_scale, (0, 255, 255), id_thick, cv2.LINE_AA)
                py = prev_y.get(tid)
                if py is not None:
                    crossed = (py < line_y <= cy) or (py > line_y >= cy)
                    if crossed and tid not in counted_ids:
                        counted_ids.add(tid)
                        counts[name] += 1
                prev_y[tid] = cy
        y0 = 40
        for k in ["bicycle", "bus", "car", "motorcycle", "truck"]:
            cv2.putText(frame, f"{k}: {counts[k]}", (20, y0), font, count_scale, (0, 255, 0), count_thick, cv2.LINE_AA)
            y0 += 35
        writer.write(frame)
    cap.release()
    writer.release()
    print("Saved:", out_path)
    print("Counts:", dict(counts))


if __name__ == "__main__":
    main()

