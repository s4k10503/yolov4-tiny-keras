import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from yolo import YOLO
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map


def main(map_mode, config):
    """
    主処理を実行する。

    Args:
        map_mode (int): 計算モード。
        config (dict): 設定値。
    """

    image_ids = open(os.path.join(config["VOCdevkit_path"], "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    prepare_output_directories(config["map_out_path"])

    class_names, _ = get_classes(config["classes_path"])

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence=config["confidence"], nms_iou=config["nms_iou"])
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            get_prediction(yolo, image_id, class_names, config)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            get_ground_truth(image_id, class_names, config)
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_voc_map(config["MINOVERLAP"], config["score_threhold"], config["map_out_path"])
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=config["map_out_path"])
        print("Get map done.")


def prepare_output_directories(map_out_path):
    """
    出力ディレクトリを準備する。

    Args:
        map_out_path (str): 出力ディレクトリパス。
    """

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))


def get_prediction(yolo, image_id, class_names, config):
    """
    予測結果を取得する。

    Args:
        yolo (YOLO): YOLOモデル。
        image_id (str): 画像ID。
        class_names (list): クラス名のリスト。
        config (dict): 設定値。
    """

    image_path = os.path.join(config["VOCdevkit_path"], "VOC2007/JPEGImages/" + image_id + ".jpg")
    image = Image.open(image_path)
    if config["map_vis"]:
        image.save(os.path.join(config["map_out_path"], "images-optional/" + image_id + ".jpg"))
    yolo.get_map_txt(image_id, image, class_names, config["map_out_path"])


def get_ground_truth(image_id, class_names, config):
    """
    グラウンドトゥルース結果を取得する。

    Args:
        image_id (str): 画像ID。
        class_names (list): クラス名のリスト。
        config (dict): 設定値。
    """

    with open(os.path.join(config["map_out_path"], "ground-truth/" + image_id + ".txt"), "w") as new_f:
        root = ET.parse(os.path.join(config["VOCdevkit_path"], "VOC2007/Annotations/" + image_id + ".xml")).getroot()
        for obj in root.findall('object'):
            process_object(obj, class_names, new_f)


def process_object(obj, class_names, new_f):
    """
    XMLオブジェクトを処理する。

    Args:
        obj (Element): XMLオブジェクト。
        class_names (list): クラス名のリスト。
        new_f (TextIOWrapper): 書き込み用のファイルオブジェクト。
    """

    difficult_flag = False
    if obj.find('difficult') is not None:
        difficult = obj.find('difficult').text
        if int(difficult) == 1:
            difficult_flag = True

    obj_name = obj.find('name').text
    if obj_name not in class_names:
        return

    bndbox = obj.find('bndbox')
    left = bndbox.find('xmin').text
    top = bndbox.find('ymin').text
    right = bndbox.find('xmax').text
    bottom = bndbox.find('ymax').text

    if difficult_flag:
        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
    else:
        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))


def get_voc_map(MINOVERLAP, score_threhold, path):
    """
    VOCマップを取得する。

    Args:
        MINOVERLAP (float): 最小オーバーラップ値。
        score_threhold (float): スコアしきい値。
        path (str): 出力ディレクトリパス。
    """

    get_map(MINOVERLAP, True, score_threhold=score_threhold, path=path)


if __name__ == "__main__":
    config = {
        "map_mode": 0,
        "classes_path": 'model_data/voc_classes.txt',
        "MINOVERLAP": 0.5,
        "confidence": 0.001,
        "nms_iou": 0.5,
        "score_threhold": 0.5,
        "map_vis": False,
        "VOCdevkit_path": 'VOCdevkit',
        "map_out_path": 'map_out'
    }

    main(config["map_mode"], config)
