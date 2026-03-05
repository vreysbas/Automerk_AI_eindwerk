import os
import shutil
from scipy.io import loadmat

# ==============================================================
# PADEN
# ==============================================================
# De basismap waar alles in staat
BASE_PATH = r"C:\Users\vreys.bas\Desktop\automerk_ai\stanford_cars"

# 1. De lijst met merknamen (Audi, BMW, etc.) uit de devkit
META_MAT = os.path.join(BASE_PATH, "devkit", "cars_meta.mat")

# 2. De labels voor de training set (uit de devkit map)
TRAIN_MAT = os.path.join(BASE_PATH, "devkit", "cars_train_annos.mat")

# 3. De labels voor de test set (staat in de hoofdmap stanford_cars)
TEST_MAT = os.path.join(BASE_PATH, "cars_test_annos_withlabels.mat")

# Waar de nieuwe, schone dataset moet komen
OUTPUT_PATH = r"C:\Users\vreys.bas\Desktop\automerk_ai\dataset_medium"

# De merken die we willen filteren
BRANDS = ["Audi", "BMW", "Mercedes"]


def create_folders():
    """Maakt de mappen aan en wast ze eerst even wit (leegmaken)."""
    print("Mappen structuur voorbereiden...")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for split in ["train", "test"]:
        for brand in BRANDS:
            path = os.path.join(OUTPUT_PATH, split, brand)
            if os.path.exists(path):
                # Map leegmaken om oude 'foute' foto's te verwijderen
                shutil.rmtree(path)
            os.makedirs(path)


def process_set(mat_path, source_folder, split_name, class_names):
    """Koppelt de labels aan de juiste foto's en kopieert ze."""
    if not os.path.exists(mat_path):
        print(f"WAARSCHUWING: Kan {mat_path} niet vinden. Sla deze set over.")
        return 0

    data = loadmat(mat_path)
    annotations = data['annotations'][0]
    gekopieerd = 0

    print(f"Bezig met filteren van {split_name} set...")
    for anno in annotations:
        # In deze bestanden heet de naam 'fname'
        fname = str(anno['fname'][0])
        # De klasse index (bijv. 1 t/m 196) omzetten naar 0-based voor de lijst
        class_id = int(anno['class'][0][0]) - 1
        full_class_name = class_names[class_id]

        for brand in BRANDS:
            # Check of 'audi', 'bmw' of 'mercedes' in de naam van de klasse staat
            if brand.lower() in full_class_name.lower():
                src_path = os.path.join(BASE_PATH, source_folder, fname)
                dest_path = os.path.join(OUTPUT_PATH, split_name, brand, fname)

                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                    gekopieerd += 1
                break
    return gekopieerd


def main():
    # Check of de belangrijkste bestanden er zijn
    if not os.path.exists(META_MAT):
        print(f"FOUT: {META_MAT} niet gevonden! Staat de 'devkit' map wel op de juiste plek?")
        return

    create_folders()

    # 1. Laad de echte namen van de 196 klassen
    meta = loadmat(META_MAT)
    class_names = [str(x[0]) for x in meta["class_names"][0]]

    # 2. Verwerk de Training foto's
    count_train = process_set(TRAIN_MAT, "cars_train", "train", class_names)

    # 3. Verwerk de Test foto's
    count_test = process_set(TEST_MAT, "cars_test", "test", class_names)

    print("-" * 40)
    print(f"Dataset succesvol opgebouwd!")
    print(f"- Audi/BMW/Mercedes in Train: {count_train}")
    print(f"- Audi/BMW/Mercedes in Test:  {count_test}")
    print(f"- Totaal aantal foto's:        {count_train + count_test}")
    print("-" * 40)
    print(f"Je kunt nu train.py gaan runnen met de map: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()