from __future__ import division

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import math

def visualize_box_mask(im, results, labels, threshold=0.5, distance_threshold=125):
    """
    Args:
        im (str/np.ndarray): path of image/np.ndarray read by cv2
        results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                        matrix element: [class, score, x_min, y_min, x_max, y_max]
                        MaskRCNN's results include 'masks': np.ndarray:
                        shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): Threshold of score.
        distance_threshold (float): Distance threshold to determine if a pig is isolated.
    Returns:
        im (PIL.Image.Image): visualized image
    """
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    elif isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    if 'masks' in results and 'boxes' in results and len(results['boxes']) > 0:
        im = draw_mask(
            im, results['boxes'], results['masks'], labels, threshold=threshold)
    if 'boxes' in results and len(results['boxes']) > 0:
        im = draw_box(im, results['boxes'], labels, threshold=threshold, distance_threshold=distance_threshold)
    if 'segm' in results:
        im = draw_segm(
            im,
            results['segm'],
            results['label'],
            results['score'],
            labels,
            threshold=threshold)
    return im


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def get_center(bbox):
    """Calculate the center of a bounding box."""
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return np.array([center_x, center_y])


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1, box2: Bounding boxes in the format [xmin, ymin, xmax, ymax]
    Returns:
        iou: Intersection over Union score.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate intersection area
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate areas of the bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def draw_box(im, np_boxes, labels, threshold=0.5, distance_threshold=600, iou_threshold=0.025):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matrix element: [class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of box confidence score
        distance_threshold (float): threshold for determining isolated pigs based on distance
        iou_threshold (float): threshold for determining isolated pigs based on IoU
    Returns:
        im (PIL.Image.Image): visualized image with isolated pigs labeled
    """
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    centers = [get_center(bbox[2:]) for bbox in np_boxes]  # Get centers of all pigs

    for i, dt in enumerate(np_boxes):
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        center = centers[i]

        # Calculate distance to the nearest pig and IoU
        min_distance = float('inf')  # Set an initially large value
        max_iou = 0  # Store maximum IoU with other pigs
        for j, other_center in enumerate(centers):
            if i != j:
                distance = calculate_distance(center, other_center)
                min_distance = min(min_distance, distance)

                # Calculate IoU between this pig and another pig
                iou = calculate_iou(bbox, np_boxes[j][2:])
                max_iou = max(max_iou, iou)

        # Check if the nearest pig is further than the distance threshold and IoU is less than iou_threshold
        is_isolated = min_distance > distance_threshold and max_iou < iou_threshold

        # Use red color for isolated pigs, default color for others
        if is_isolated:
            color = (255, 0, 0)  # Red for isolated pigs
        else:
            if clsid not in clsid2color:
                clsid2color[clsid] = color_list[clsid]
            color = tuple(clsid2color[clsid])

        # Draw bounding box
        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
                  'right_bottom:[{:.2f},{:.2f}]'.format(
                      int(clsid), score, xmin, ymin, xmax, ymax))
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=draw_thickness,
                fill=color)

            # If isolated, add "isolated" label with red background
            if is_isolated:
                text = "isolated"  # Label for isolated pigs
                l, t, r, b = draw.textbbox((0, 0), text)
                tw, th = r - l, b - t
                label_x = xmin
                label_y = ymax

                # Draw red rectangle for the background of the "isolated" label
                draw.rectangle(
                    [(label_x, label_y), (label_x + tw + 2, label_y + th + 2)],
                    fill="red")

                # Draw the "isolated" label text in white
                draw.text((label_x + 1, label_y + 1), text, fill=(255, 255, 255))

        # Draw label for confidence score and class name; change to red if isolated
        text = "{} {:.4f}".format('pig', score)
        l, t, r, b = draw.textbbox((0, 0), text)
        tw, th = r - l, b - t
        if is_isolated:
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill="red")  # Red for isolated pigs
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))  # Text in white
        else:
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill="green")  # Green for others
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))  # Text in white

    return im




def draw_mask(im, np_boxes, np_masks, labels, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
            matrix element: [class, score, x_min, y_min, x_max, y_max]
        np_masks (np.ndarray): shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of mask confidence score
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_color_map_list(len(labels))
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype('float32')
    clsid2color = {}
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]
    np_masks = np_masks[expect_boxes, :, :]
    im_h, im_w = im.shape[:2]
    np_masks = np_masks[:, :im_h, :im_w]
    for i in range(len(np_masks)):
        clsid, score = int(np_boxes[i][0]), np_boxes[i][1]
        mask = np_masks[i]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype('uint8'))


def draw_segm(im,
              np_segms,
              np_label,
              np_score,
              labels,
              threshold=0.5,
              alpha=0.7):
    """
    Draw segmentation on image
    """
    mask_color_id = 0
    w_ratio = .4
    color_list = get_color_map_list(len(labels))
    im = np.array(im).astype('float32')
    clsid2color = {}
    np_segms = np_segms.astype(np.uint8)
    for i in range(np_segms.shape[0]):
        mask, score, clsid = np_segms[i], np_score[i], np_label[i]
        if score < threshold:
            continue

        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        idx0 = np.minimum(idx[0], im.shape[0] - 1)
        idx1 = np.minimum(idx[1], im.shape[1] - 1)
        im[idx0, idx1, :] *= 1.0 - alpha
        im[idx0, idx1, :] += alpha * color_mask
        sum_x = np.sum(mask, axis=0)
        x = np.where(sum_x > 0.5)[0]
        sum_y = np.sum(mask, axis=1)
        y = np.where(sum_y > 0.5)[0]
        x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
        cv2.rectangle(im, (x0, y0), (x1, y1),
                      tuple(color_mask.astype('int32').tolist()), 1)
        bbox_text = '%s %.2f' % (labels[clsid], score)
        t_size = cv2.getTextSize(bbox_text, 0, 0.3, thickness=1)[0]
        cv2.rectangle(im, (x0, y0), (x0 + t_size[0], y0 - t_size[1] - 3),
                      tuple(color_mask.astype('int32').tolist()), -1)
        cv2.putText(
            im,
            bbox_text, (x0, y0 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, (0, 0, 0),
            1,
            lineType=cv2.LINE_AA)
    return Image.fromarray(im.astype('uint8'))
