import argparse
import os
from pathlib import Path
import csv
from typing import List, Tuple, Optional

import cv2

# Ultralytics YOLO (works for YOLOv8 and many YOLOv10 forks that keep Ultralytics API)
from ultralytics import YOLO


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iuatld_grade_from_count(afb_count: int) -> str:
    """
    Simple placeholder grading based on count.
    Adjust to your protocol (per field / per slide) if needed.
    """
    if afb_count <= 0:
        return "Negative"
    if 1 <= afb_count <= 9:
        return "Scanty"
    if 10 <= afb_count <= 99:
        return "TB 1+"
    if 100 <= afb_count <= 999:
        return "TB 2+"
    return "TB 3+"


def list_images(source: str) -> List[Path]:
    p = Path(source)
    if p.is_file():
        if p.suffix.lower() in IMG_EXT:
            return [p]
        raise ValueError(f"File is not an image: {p}")
    if p.is_dir():
        imgs = [x for x in p.rglob("*") if x.suffix.lower() in IMG_EXT]
        if not imgs:
            raise ValueError(f"No images found in folder: {p}")
        return sorted(imgs)
    raise ValueError(f"Source not found: {source}")


def draw_boxes(
    img_bgr,
    boxes_xyxy: List[Tuple[int, int, int, int]],
    confs: List[float],
    cls_names: List[str],
    classes: List[int],
):
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, classes):
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{cls_names[cls_id]} {conf:.2f}" if cls_id < len(cls_names) else f"id{cls_id} {conf:.2f}"
        cv2.putText(
            img_bgr,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return img_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to .pt weights (e.g., yolov10i.pt)")
    ap.add_argument("--source", type=str, required=True, help="Image file or folder")
    ap.add_argument("--out", type=str, default="runs/detect", help="Output directory")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (if supported by model)")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--device", type=str, default="", help="cuda / cpu / '' (auto)")
    ap.add_argument("--class_id", type=int, default=None, help="Optional: only count this class id as AFB")
    ap.add_argument("--save_csv", action="store_true", help="Save CSV summary")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.source)

    model = YOLO(args.weights)
    cls_names = model.names if hasattr(model, "names") else {}

    csv_rows = []
    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        # Ultralytics inference
        results = model.predict(
            source=img_bgr,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device if args.device != "" else None,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes

        afb_boxes = []
        afb_confs = []
        afb_classes = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), cf, cid in zip(xyxy, confs, clss):
                if args.class_id is not None and cid != args.class_id:
                    continue
                afb_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                afb_confs.append(float(cf))
                afb_classes.append(int(cid))

        afb_count = len(afb_boxes)
        grade = iuatld_grade_from_count(afb_count)

        annotated = draw_boxes(img_bgr.copy(), afb_boxes, afb_confs, list(cls_names.values()) if isinstance(cls_names, dict) else cls_names, afb_classes)

        # Put summary text
        cv2.putText(
            annotated,
            f"AFB count: {afb_count} | Grade: {grade}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        save_path = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(save_path), annotated)

        csv_rows.append([str(img_path), afb_count, grade, str(save_path)])

        print(f"[OK] {img_path.name} -> count={afb_count}, grade={grade}")

    if args.save_csv:
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "afb_count", "grade", "output_image"])
            w.writerows(csv_rows)
        print(f"[OK] CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
