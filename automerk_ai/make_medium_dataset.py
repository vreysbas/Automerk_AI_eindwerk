import os
import shutil
from scipy.io import loadmat

# Waar je Stanford Cars staat
BASE = r"C:\Users\vreys.bas\Desktop\automerk_ai\dataset\StanfordCars-Dataset\stanford_cars"


# Waar we de nieuwe dataset gaan maken (die jouw PyTorch code snapt)
OUT = r"C:\Users\vreys.bas\Desktop\automerk_ai\automerk_ai\dataset_medium"

TRAIN_DIR = os.path.join(BASE, "cars_train")
TEST_DIR  = os.path.join(BASE, "cars_test")
DEVKIT    = os.path.join(BASE, "devkit")

META_PATH  = os.path.join(DEVKIT, "cars_meta.mat")
TRAIN_ANNO = os.path.join(BASE, "cars_annos.mat")
TEST_ANNO  = os.path.join(BASE, "cars_test_annos_withlabels.mat")

TARGET_BRANDS = ["audi", "bmw", "mercedes"]

MAX_TRAIN_PER_BRAND = 100
MAX_TEST_PER_BRAND  = 30


def ensure_dirs():
    for split in ["train", "test"]:
        for b in TARGET_BRANDS:
            os.makedirs(os.path.join(OUT, split, b), exist_ok=True)


def get_class_names():
    meta = loadmat(META_PATH)
    class_names = meta["class_names"][0]
    names = []
    for i in range(len(class_names)):
        names.append(str(class_names[i][0]))
    return names


def read_annotations(path):
    data = loadmat(path)
    annos = data["annotations"][0]
    items = []
    for a in annos:
        fname = str(a["fname"][0])
        class_id = int(a["class"][0][0])  # 1..196
        items.append((fname, class_id))
    return items


def brand_from_classname(classname: str):
    low = classname.lower()
    for b in TARGET_BRANDS:
        if b in low:
            return b
    return None


def copy_split(items, class_names, src_dir, split, max_per_brand):
    counters = {b: 0 for b in TARGET_BRANDS}

    for fname, class_id in items:
        classname = class_names[class_id - 1]
        brand = brand_from_classname(classname)
        if brand is None:
            continue
        if counters[brand] >= max_per_brand:
            continue

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(OUT, split, brand, fname)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            counters[brand] += 1

    print(f"{split} counters:", counters)


def main():
    # checks
    for p in [BASE, TRAIN_DIR, TEST_DIR, DEVKIT, META_PATH, TRAIN_ANNO, TEST_ANNO]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Niet gevonden: {p}")

    ensure_dirs()
    class_names = get_class_names()

    train_items = read_annotations(TRAIN_ANNO)
    test_items  = read_annotations(TEST_ANNO)

    copy_split(train_items, class_names, TRAIN_DIR, "train", MAX_TRAIN_PER_BRAND)
    copy_split(test_items,  class_names, TEST_DIR,  "test",  MAX_TEST_PER_BRAND)

    print("✅ Klaar! Dataset staat in:", OUT)


if __name__ == "__main__":
    main()
