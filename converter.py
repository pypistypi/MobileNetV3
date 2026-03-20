import os
import xml.etree.ElementTree as ET

# --- НАСТРОЙКИ ---
CLASSES = ["left_eye", "right_eye"]  # Убедитесь, что порядок такой же, как в MakeSense!
INPUT_DIRS = [
    'datasets/eyes/labels/train',
    'datasets/eyes/labels/val'
]


# -----------------

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


for directory in INPUT_DIRS:
    for file in os.listdir(directory):
        if not file.endswith('.xml'): continue

        path = os.path.join(directory, file)
        tree = ET.parse(path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        out_file = open(path.replace('.xml', '.txt'), 'w')
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in CLASSES: continue
            cls_id = CLASSES.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")
        out_file.close()
        print(f"Конвертирован: {file}")
