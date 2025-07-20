import os

dataset_path = 'datasaet_aksara jawa'

for class_name in sorted(os.listdir(dataset_path)):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        total = len(os.listdir(class_dir))
        print(f"{class_name}: {total} gambar")
