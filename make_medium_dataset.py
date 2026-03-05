import os
import shutil
from scipy.io import loadmat

BASE_PATH = r"C:\Users\vreys.bas\Desktop\automerk_ai\dataset\StanfordCars-Dataset\stanford_cars"
OUTPUT_PATH = r"C:\Users\vreys.bas\Desktop\automerk_ai\dataset_medium"

BRANDS = {
    "Audi": "audi",
    "BMW": "bmw",
    "Mercedes": "mercedes"
}


def create_folders():
    for split in ["train", "test"]:
        for brand in BRANDS.keys():
            path = os.path.join(OUTPUT_PATH, split, brand)
            os.makedirs(path, exist_ok=True)


def main():
    # create_folders()

    data = loadmat(os.path.join(BASE_PATH, "cars_annos.mat"))
    annotations = data["annotations"][0]
    class_names = [str(x[0]) for x in data["class_names"][0]]
    print("names:", class_names)



if __name__ == "__main__":
    main()