import os
import shutil
import scipy.io

# Laad de metadata
data = scipy.io.loadmat("stanford_cars/cars_annos.mat")
annotations = data["annotations"][0]

# Pad waar je de medium dataset wilt
train_path = "dataset_medium/train"
test_path = "dataset_medium/test"

# Maak de mappen aan als ze nog niet bestaan
classes = ["audi", "bmw", "mercedes"]
for c in classes:
    os.makedirs(os.path.join(train_path, c), exist_ok=True)
    os.makedirs(os.path.join(test_path, c), exist_ok=True)

# Loop door alle annotations
for a in annotations:
    # relatief pad naar de afbeelding
    rel_path = a["relative_im_path"][0]
    full_path = os.path.join("stanford_cars", rel_path)

    # merk van de auto
    class_name = a["class"][0].lower()
    if class_name not in classes:
        continue  # sla alle andere merken over

    # bepaal of train of test
    is_test = a["test"][0][0]
    dest_dir = test_path if is_test else train_path

    # kopieer afbeelding
    shutil.copy(full_path, os.path.join(dest_dir, class_name))