#!/usr/bin/env python
# coding: utf-8
import argparse
import glob
import json
import os
import os.path as osp
import shutil
import cv2
import numpy as np
from tqdm import tqdm

label_to_num = {}
categories_list = []
labels_list = []

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def images_labelme(data, num, img_folder):
    image = {}
    
    # 如果有 imageHeight 和 imageWidth 就用，没有的话就从图片文件获取
    if 'imageHeight' in data and 'imageWidth' in data:
        image['height'] = data['imageHeight']
        image['width'] = data['imageWidth']
    else:
        img_path = osp.join(img_folder, data['imagePath'])
        img = cv2.imread(img_path)
        if img is not None:
            image['height'] = img.shape[0]
            image['width'] = img.shape[1]
        else:
            raise FileNotFoundError(f"图像文件 {img_path} 未找到或无法读取")
    
    image['id'] = num + 1
    image['file_name'] = osp.basename(data['imagePath'])
    return image


def categories(label, labels_list):
    category = {}
    category['supercategory'] = 'component'
    category['id'] = len(labels_list) + 1
    category['name'] = label
    return category


def annotations_rectangle_from_boxes(boxes, label, image_num, object_num, label_to_num):
    annotation = {}
    x1, y1, x2, y2 = boxes
    annotation['segmentation'] = [[x1, y1, x2, y1, x2, y2, x1, y2]]  # 用四点构成矩形
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_num + 1
    annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    annotation['area'] = (x2 - x1) * (y2 - y1)
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = object_num + 1
    return annotation


def annotations_rectangle_from_points(points, label, image_num, object_num, label_to_num):
    """根据 points 计算 bbox"""
    annotation = {}
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # 计算最小和最大 x, y 以形成包围框
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    
    # 创建 segmentation 和 bbox
    annotation['segmentation'] = [list(np.asarray(points).flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_num + 1
    annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    annotation['area'] = (x2 - x1) * (y2 - y1)
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = object_num + 1
    return annotation


def deal_json(ds_type, img_path, json_path):
    data_coco = {}
    images_list = []
    annotations_list = []
    image_num = -1
    object_num = -1

    for img_file in os.listdir(img_path):
        img_label = os.path.splitext(img_file)[0]
        if img_file.split('.')[-1] not in ['bmp', 'jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG']:
            continue

        label_file = osp.join(json_path, img_label + '.json')
        print('Generating dataset from:', label_file)
        image_num += 1
        
        with open(label_file) as f:
            data = json.load(f)
            
            if ds_type == 'labelme':
                images_list.append(images_labelme(data, image_num, img_path))
            
            if 'shape' in data:
                for shapes in data['shape']:
                    label = shapes['label']
                    
                    # 如果有 boxes 就使用它
                    if 'boxes' in shapes and shapes['boxes'] is not None:
                        boxes = shapes['boxes']
                        object_num += 1
                        if label not in labels_list:
                            categories_list.append(categories(label, labels_list))
                            labels_list.append(label)
                            label_to_num[label] = len(labels_list)
                        annotations_list.append(
                            annotations_rectangle_from_boxes(boxes, label, image_num, object_num, label_to_num)
                        )
                    
                    # 如果没有 boxes，但有 points，就根据 points 计算 bbox
                    elif 'points' in shapes and shapes['points'] is not None:
                        points = shapes['points']
                        object_num += 1
                        if label not in labels_list:
                            categories_list.append(categories(label, labels_list))
                            labels_list.append(label)
                            label_to_num[label] = len(labels_list)
                        annotations_list.append(
                            annotations_rectangle_from_points(points, label, image_num, object_num, label_to_num)
                        )
    
    data_coco['images'] = images_list
    data_coco['categories'] = categories_list
    data_coco['annotations'] = annotations_list
    return data_coco


def voc_get_label_anno(ann_dir_path, ann_ids_path, labels_path):
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))

    with open(ann_ids_path, 'r') as f:
        ann_ids = [lin.strip().split(' ')[-1] for lin in f.readlines()]

    ann_paths = []
    for aid in ann_ids:
        if aid.endswith('xml'):
            ann_path = os.path.join(ann_dir_path, aid)
        else:
            ann_path = os.path.join(ann_dir_path, aid + '.xml')
        ann_paths.append(ann_path)

    return dict(zip(labels_str, labels_ids)), ann_paths


def voc_xmls_to_cocojson(annotation_paths, label2id, output_dir, output_file):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # bounding box start id
    im_id = 0
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # 读取 xml 标注
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = voc_get_image_info(ann_root, im_id)
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = voc_get_coco_annotation(obj=obj, label2id=label2id)
            ann.update({'image_id': im_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id += 1
        im_id += 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)
    output_file = os.path.join(output_dir, output_file)
    with open(output_file, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_type', help='the type of dataset, can be `voc`, `labelme`, `cityscape`, or `widerface`')
    parser.add_argument('--json_input_dir', help='input annotated directory')
    parser.add_argument('--image_input_dir', help='image directory')
    parser.add_argument('--output_dir', help='output dataset directory', default='./')
    parser.add_argument('--train_proportion', type=float, default=0.8, help='Proportion of training dataset')
    parser.add_argument('--val_proportion', type=float, default=0.1, help='Proportion of validation dataset')
    parser.add_argument('--test_proportion', type=float, default=0.1, help='Proportion of test dataset')
    args = parser.parse_args()

    try:
        assert abs(args.train_proportion + args.val_proportion + args.test_proportion - 1.0) < 1e-5
    except AssertionError:
        print('The sum of training, validation, and test proportions must equal 1!')
        os._exit(0)

    total_num = len(glob.glob(osp.join(args.json_input_dir, '*.json')))
    train_num = int(total_num * args.train_proportion)
    val_num = int(total_num * args.val_proportion)
    test_num = total_num - train_num - val_num

    # 划分数据集并创建对应文件夹
    files = os.listdir(args.image_input_dir)
    train_files = files[:train_num]
    val_files = files[train_num:train_num + val_num]
    test_files = files[train_num + val_num:]

    output_dirs = ['train', 'val', 'test']
    for folder in output_dirs:
        os.makedirs(os.path.join(args.output_dir, folder), exist_ok=True)

    for phase, phase_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for file in phase_files:
            shutil.copy(os.path.join(args.image_input_dir, file), os.path.join(args.output_dir, phase))

        # 生成 COCO 格式标注
        phase_json_path = osp.join(args.output_dir + '/annotations', f'instances_{phase}.json')
        phase_data_coco = deal_json(args.dataset_type, osp.join(args.output_dir, phase), args.json_input_dir)
        json.dump(phase_data_coco, open(phase_json_path, 'w'), indent=4, cls=MyEncoder)


if __name__ == '__main__':
    main()
