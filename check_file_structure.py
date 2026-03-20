import os

def check_structure(base_path):
    output_file = "file_structure_check.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Проверка структуры в: {base_path}\n\n")

        subdirs = ["images", "masks_i", "masks_p"]
        for subdir in subdirs:
            current_path = os.path.join(base_path, subdir)
            f.write(f"--- Содержимое папки {current_path} ---\n")
            if os.path.exists(current_path) and os.path.isdir(current_path):
                files = os.listdir(current_path)
                if files:
                    for file_name in files:
                        f.write(f"  - {file_name}\n")
                else:
                    f.write("  (Папка пуста)\n")
            else:
                f.write("  (Папка не найдена или не является директорией)\n")
            f.write("\n")
    print(f"Результаты записаны в {output_file}")

if __name__ == "__main__":
    # Укажите путь к вашей папке big_dataset
    # ВНИМАНИЕ: Для Windows пути нужно указывать с двойными обратными слешами или с обычными слешами
    # Например: "C:\\Users\\admin\\PycharmProjects\\MobileNetV3\\datasets\\big_dataset"
    # Или: "C:/Users/admin/PycharmProjects/MobileNetV3/datasets/big_dataset"
    # Я использую относительный путь, предполагая, что скрипт запускается из MobileNetV3
    base_dataset_path = "datasets/big_dataset"
    check_structure(base_dataset_path)
