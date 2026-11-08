import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

BASE_DATASET_DIR = "/workspace/data/bdd"
OUTPUT_BASE_DIR = "/workspace/data/BDD-C"
SPLITS = ["train", "val"]

NUM_WORKERS = max(1, min(32, (os.cpu_count() or 4) - 1))
CHUNKSIZE = 31

TASKS = {
    "task_1_clear_day": lambda weather, timeofday: timeofday == "daytime"
    and weather in {"clear", "partly cloudy"},
    "task_2_adverse_day": lambda weather, timeofday: timeofday == "daytime"
    and weather in {"rainy", "snowy", "foggy", "overcast"},
    "task_3_dawn_dusk": lambda weather, timeofday: timeofday == "dawn/dusk",
    "task_4_clear_night": lambda weather, timeofday: timeofday == "night"
    and weather in {"clear", "partly cloudy"},
    "task_5_adverse_night": lambda weather, timeofday: timeofday == "night"
    and weather in {"rainy", "snowy", "foggy", "overcast"},
}

OUTPUT_IMAGE_SIZE = (224, 224)
JPEG_QUALITY = 95
MANIFEST_DIRNAME = "manifests"
METADATA_FILENAME = "metadata.json"
RESAMPLE = Image.Resampling.BILINEAR
MANIFEST_ATTRIBUTE_KEYS = ("weather", "timeofday", "scene")


def get_image_name(item, json_filename):
    """Return the matching image filename for an annotation file."""
    img_name = item.get("name", json_filename.replace(".json", ".jpg"))
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_name += ".jpg"
    return img_name


def get_valid_boxes(item, image_width, image_height):
    """Extract valid clamped boxed objects from the first frame."""
    frames = item.get("frames", [])
    if not frames:
        return []

    objects = frames[0].get("objects", [])
    valid_boxes = []

    for obj in objects:
        box = obj.get("box2d")
        if not box:
            continue

        x1 = max(0, min(image_width, int(box.get("x1", 0))))
        y1 = max(0, min(image_height, int(box.get("y1", 0))))
        x2 = max(0, min(image_width, int(box.get("x2", 0))))
        y2 = max(0, min(image_height, int(box.get("y2", 0))))

        if x2 <= x1 or y2 <= y1:
            continue

        category = obj.get("category")
        if not category:
            continue

        valid_boxes.append({
            "object_id": obj.get("id"),
            "category": str(category),
            "bbox": [x1, y1, x2, y2],
        })

    return valid_boxes


def create_output_directories(base_dir):
    output_root = Path(base_dir)
    for split in SPLITS:
        for task_name in TASKS:
            (output_root / split / task_name).mkdir(parents=True, exist_ok=True)
        (output_root / MANIFEST_DIRNAME / split).mkdir(parents=True, exist_ok=True)
    return output_root


def resolve_task(weather, timeofday):
    for task_name, condition_func in TASKS.items():
        if condition_func(weather, timeofday):
            return task_name
    return None


def get_manifest_attributes(item):
    """Extract image-level BDD100K attributes preserved in output manifests."""
    attrs = item.get("attributes", {}) or {}
    return {
        attr_name: str(attrs.get(attr_name, "undefined"))
        for attr_name in MANIFEST_ATTRIBUTE_KEYS
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_metadata(output_base_dir, split_counts, categories, attribute_values):
    payload = {
        "format": "jpeg_manifest",
        "image_shape": [OUTPUT_IMAGE_SIZE[1], OUTPUT_IMAGE_SIZE[0], 3],
        "class_names": sorted(categories),
        "metric_groups": {
            attr_name: sorted(values)
            for attr_name, values in sorted(attribute_values.items())
        },
        "splits": {
            split: {"task_counts": dict(sorted(task_counts.items()))}
            for split, task_counts in split_counts.items()
        },
    }

    metadata_path = Path(output_base_dir) / MANIFEST_DIRNAME / METADATA_FILENAME
    write_json(metadata_path, payload)
    print(f"Wrote metadata: {metadata_path}")


def process_annotation_file(args):
    split_name, json_filename, input_img_dir, input_label_dir, output_base_dir = args
    json_path = os.path.join(input_label_dir, json_filename)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "status": "error",
            "error": f"JSON load failed for {json_filename}: {e}",
            "records": [],
        }

    item = data[0] if isinstance(data, list) else data
    image_attributes = get_manifest_attributes(item)
    weather = image_attributes["weather"]
    timeofday = image_attributes["timeofday"]

    task_name = resolve_task(weather, timeofday)
    if task_name is None:
        return {"status": "skip", "records": []}

    img_name = get_image_name(item, json_filename)
    img_path = os.path.join(input_img_dir, img_name)
    if not os.path.exists(img_path):
        return {"status": "missing", "records": []}

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            valid_boxes = get_valid_boxes(item, *img.size)
            if not valid_boxes:
                return {"status": "no_box", "records": []}

            records = []
            stem = Path(json_filename).stem
            for object_index, box_info in enumerate(valid_boxes):
                x1, y1, x2, y2 = box_info["bbox"]
                cropped_img = img.crop((x1, y1, x2, y2))
                resized_img = cropped_img.resize(OUTPUT_IMAGE_SIZE, RESAMPLE)

                output_rel = Path(split_name) / task_name / f"{stem}_{object_index:03d}.jpg"
                output_path = Path(output_base_dir) / output_rel
                resized_img.save(output_path, format="JPEG", quality=JPEG_QUALITY, optimize=False)

                records.append({
                    "image": str(output_rel),
                    "label": box_info["category"],
                    "category": box_info["category"],
                    **image_attributes,
                    "task": task_name,
                    "source_image": img_name,
                    "source_annotation": f"{split_name}/{json_filename}",
                    "source_object_id": box_info["object_id"],
                    "bbox": box_info["bbox"],
                })
    except Exception as e:
        return {
            "status": "error",
            "error": f"Image processing failed for {img_name}: {e}",
            "records": [],
        }

    return {
        "status": "success",
        "task": task_name,
        "records": records,
        "count": len(records),
    }


def process_split(split_name, input_img_dir, input_label_dir, output_base_dir):
    print("\n" + "=" * 40)
    print(f"PROCESSING SPLIT: {split_name.upper()}")
    print("=" * 40)

    if not os.path.exists(input_img_dir):
        print(f"SKIPPED: Image directory not found at {input_img_dir}")
        return None
    if not os.path.exists(input_label_dir):
        print(f"SKIPPED: Label directory not found at {input_label_dir}")
        return None

    json_files = sorted(f for f in os.listdir(input_label_dir) if f.endswith(".json"))
    print(f"Found {len(json_files)} annotation files.")
    print(f"Using {NUM_WORKERS} worker processes")

    success_count = 0
    skip_count = 0
    missing_img_count = 0
    no_box_count = 0
    error_count = 0
    categories = set()
    attribute_values = defaultdict(set)
    task_counts = defaultdict(int)
    manifests = defaultdict(list)

    worker_args = [
        (split_name, json_filename, input_img_dir, input_label_dir, output_base_dir)
        for json_filename in json_files
    ]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = executor.map(process_annotation_file, worker_args, chunksize=CHUNKSIZE)
        with tqdm(total=len(json_files), desc=f"Cropping {split_name}", unit="img") as pbar:
            for result in results:
                status = result.get("status")
                if status == "success":
                    task_name = result["task"]
                    records = result["records"]
                    manifests[task_name].extend(records)
                    count = int(result.get("count", len(records)))
                    success_count += count
                    task_counts[task_name] += count
                    for record in records:
                        categories.add(record["category"])
                        for attr_name in MANIFEST_ATTRIBUTE_KEYS:
                            attribute_values[attr_name].add(record[attr_name])
                elif status == "skip":
                    skip_count += 1
                elif status == "missing":
                    missing_img_count += 1
                elif status == "no_box":
                    no_box_count += 1
                else:
                    error_count += 1
                    error_msg = result.get("error")
                    if error_msg:
                        tqdm.write(error_msg)

                pbar.update(1)
                if pbar.n % 250 == 0 or pbar.n == pbar.total:
                    pbar.set_postfix(
                        Success=success_count,
                        Skip=skip_count,
                        Miss=missing_img_count,
                        NoBox=no_box_count,
                        Err=error_count,
                    )

    manifest_split_dir = Path(output_base_dir) / MANIFEST_DIRNAME / split_name
    for task_name in TASKS:
        manifest_path = manifest_split_dir / f"{task_name}.json"
        write_json(manifest_path, sorted(manifests[task_name], key=lambda record: record["image"]))

    print(f"\n--- {split_name.upper()} SUMMARY ---")
    print(f"Successfully saved: {success_count} cropped images")
    print(f"Skipped (didn't match 5 tasks): {skip_count} images")
    print(f"Missing image files: {missing_img_count}")
    print(f"Images with no valid boxes: {no_box_count}")
    print(f"Processing errors: {error_count}")
    print(f"Output directory: {Path(output_base_dir) / split_name}")
    print(f"Manifest directory: {manifest_split_dir}")

    return {
        "task_counts": task_counts,
        "categories": categories,
        "attribute_values": attribute_values,
        "success_count": success_count,
        "skip_count": skip_count,
        "missing_img_count": missing_img_count,
        "no_box_count": no_box_count,
        "error_count": error_count,
    }


if __name__ == "__main__":
    output_root = create_output_directories(OUTPUT_BASE_DIR)
    split_counts = {}
    all_categories = set()
    all_attribute_values = defaultdict(set)

    for split in SPLITS:
        img_dir = os.path.join(BASE_DATASET_DIR, "images", split)
        lbl_dir = os.path.join(BASE_DATASET_DIR, "labels", split)
        summary = process_split(split, img_dir, lbl_dir, output_root)
        if summary is not None:
            split_counts[split] = summary["task_counts"]
            all_categories.update(summary["categories"])
            for attr_name, values in summary["attribute_values"].items():
                all_attribute_values[attr_name].update(values)

    if split_counts:
        write_metadata(
            OUTPUT_BASE_DIR,
            split_counts=split_counts,
            categories=all_categories,
            attribute_values=all_attribute_values,
        )

    print("\n" + "=" * 40)
    print("ALL SPLITS PROCESSED SUCCESSFULLY!")
    print("=" * 40)
