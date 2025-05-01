import argparse
import os
import json
from PIL import Image
import supervision as sv
from ultralytics import YOLOE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="yoloe-v8l-seg.pt", help="Model checkpoint path")
    parser.add_argument("--names", nargs="+", default=["person"], help="Prompt/class names")
    parser.add_argument("--output", type=str, help="Path to save annotated image")
    parser.add_argument("--json-out", type=str, help="Path to save detection metadata")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cpu/cuda)")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold (default: 0.2)")
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.source)
        args.output = f"{base}-output{ext}"

    if not args.json_out:
        args.json_out = args.output.replace(".jpg", ".json")

    image = Image.open(args.source).convert("RGB")

    # Load model and set text prompts
    model = YOLOE(args.checkpoint)
    model.to(args.device)
    model.set_classes(args.names, model.get_text_pe(args.names))

    # Run inference
    results = model.predict(image, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    # Filter by confidence
    detections = detections[detections.confidence > args.conf]

    if len(detections) == 0:
        print("🔍 Detected 0 objects above confidence threshold")
        image.save(args.output)
        with open(args.json_out, "w") as f:
            json.dump([], f, indent=2)
        print(f"🖼️ Annotated image saved to: {args.output}")
        print(f"📦 Empty metadata saved to: {args.json_out}")
        return

    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh)

    # Safely get label using class_id index into prompt list
    labels = [
        f"{args.names[detections.class_id[i]] if detections.class_id[i] < len(args.names) else 'unknown'} {detections.confidence[i]:.2f}"
        for i in range(len(detections))
    ]

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.4
    ).annotate(scene=annotated_image, detections=detections)

    annotated_image = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        thickness=thickness
    ).annotate(scene=annotated_image, detections=detections)

    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale,
        smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    annotated_image.save(args.output)
    print(f"🖼️ Annotated image saved to: {args.output}")

    metadata = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        metadata.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(detections.confidence[i]),
            "class_id": int(detections.class_id[i]),
            "class_name": detections.class_name[i] if hasattr(detections, "class_name") else args.names[detections.class_id[i]]
        })

    with open(args.json_out, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📦 Metadata saved to: {args.json_out}")


if __name__ == "__main__":
    main()