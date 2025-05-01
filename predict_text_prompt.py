import argparse
import os
import json
from PIL import Image
import supervision as sv
from ultralytics import YOLOE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, default="yoloe-v8l-seg.pt", help="Path or ID of the model checkpoint")
    parser.add_argument("--names", nargs="+", default=["person"], help="List of class names to set for the model")
    parser.add_argument("--output", type=str, required=True, help="Path to save the annotated image")
    parser.add_argument("--json-out", type=str, help="Optional: Path to save detection metadata JSON")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve fallback JSON output path
    metadata_path = args.json_out if args.json_out else args.output.replace(".jpg", ".json")

    # Load image
    image = Image.open(args.source).convert("RGB")

    # Load YOLOE model and apply prompt-based labels
    model = YOLOE(args.checkpoint)
    model.to(args.device)
    model.set_classes(args.names, model.get_text_pe(args.names))

    # Run inference
    results = model.predict(image, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    print(f"🔍 Detected {len(detections.xyxy)} objects")

    # Generate labels
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections.get("class_name", []), detections.confidence)
    ]

    # Annotate image
    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh)

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(annotated_image, detections)
    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True).annotate(annotated_image, detections, labels)

    # Save annotated image
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    annotated_image.save(args.output)
    print(f"✅ Annotated image saved to: {args.output}")

    # Save metadata
    metadata = []
    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i]
        metadata.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(detections.confidence[i]),
            "class_id": int(detections.class_id[i]),
            "class_name": detections.get("class_name", [""])[i]  # Safe fallback
        })

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()