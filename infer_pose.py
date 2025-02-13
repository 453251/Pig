import os
import yaml
import glob
import json
from pathlib import Path
from functools import reduce
import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config, create_predictor
import sys
from PIL import Image, ImageDraw, ImageFile

# 导入Paddle的其他工具和自定义模块
from benchmark_utils import PaddleInferBenchmark
from preprocess import preprocess, Resize, NormalizeImage, Permute, Pad, decode_image
from visualize import visualize_box_mask
from utils import argsparser, Timer, get_current_memory_mb, multiclass_nms, coco_clsid2catid

import time
from datetime import datetime, timedelta

# 字典用于存储每只猪的最后位置和它被检测为不动的时间
pig_position_tracker = {}

# 定义投喂时间
feeding_times = [(7, 0), (12, 0), (17, 30)]  # 7:00、12:00 和 17:30
movement_threshold_seconds = 10  # 10秒内未移动则判定为静止
time_window = 30  # 半小时的时间窗口，单位：分钟
distance_threshold = 50  # 用于位置匹配的距离阈值

from datetime import time


def is_within_feeding_window(current_time, feeding_times, time_window):
    """
    检查当前时间是否在投喂时间的前后半小时内
    Args:
        current_time (datetime.time): 当前时间
        feeding_times (list): 设定的投喂时间列表，格式为 [(hour, minute), ...]
        time_window (int): 时间窗口（分钟）
    Returns:
        bool: 如果当前时间在投喂时间的时间窗口内，则返回 True
    """
    current_datetime = datetime.combine(datetime.today(), current_time)
    for hour, minute in feeding_times:
        feeding_time = datetime.combine(datetime.today(), time(hour, minute))  # 正确使用 datetime.time()
        end_time = feeding_time + timedelta(minutes=time_window)
        if current_datetime <= end_time:
            return True
    return False


def assign_pig_id(current_center, last_positions):
    """
    根据中心点位置分配猪的 ID
    Args:
        current_center (tuple): 当前猪的中心点 (x, y)
        last_positions (dict): 记录上一帧中所有猪的中心点及其对应的 ID
    Returns:
        str: 分配的猪的 ID
    """
    for pig_id, last_center in last_positions.items():
        distance = np.linalg.norm(np.array(current_center) - np.array(last_center))
        if distance < distance_threshold:
            return pig_id  # 如果距离小于阈值，认为是同一只猪
    # 如果没有找到匹配的，返回新 ID
    new_id = f"pig_{len(last_positions) + 1}"
    return new_id


# 全局支持的模型字典
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'YOLOX', 'YOLOF', 'YOLOv5', 'RTMDet', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'DETR'
}


class Detector(object):
    def __init__(self,
                 model_dir,  # 目标检测模型路径
                 pose_model_dir,  # 姿态检测模型路径
                 label_file,  # 姿态标签文件路径
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir='output',
                 threshold=0.5,
                 delete_shuffle_pass=False):
        self.pred_config = self.set_config(model_dir)
        self.pose_predictor_exe, self.pose_inference_program, self.feed_target_names, self.fetch_targets = self.load_pose_predictor(
            pose_model_dir)  # 加载姿态检测模型
        self.label_list = self.load_labels(label_file)  # 加载姿态标签
        self.predictor, self.config = load_predictor(
            model_dir,
            self.pred_config.arch,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            delete_shuffle_pass=delete_shuffle_pass)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold

    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def load_pose_predictor(self, model_dir):
        """加载姿态检测模型"""
        # place = paddle.CPUPlace()
        place = paddle.CUDAPlace(0)
        exe = paddle.fluid.Executor(place)
        [inference_program, feed_target_names, fetch_targets] = paddle.fluid.io.load_inference_model(
            dirname=model_dir, executor=exe)
        return exe, inference_program, feed_target_names, fetch_targets

    def load_labels(self, label_file):
        """加载姿态标签"""
        with open(label_file, 'r') as f:
            load_dict = json.load(f)
            label_list = [load_dict['class_detail'][i]['class_name'] for i in range(load_dict['all_class_sum'])]
        return label_list

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def pose_predict(self, exe, inference_program, feed_target_names, cropped_img):
        """姿态检测模型推理"""
        # 姿态检测图像预处理
        pose_input = cv2.resize(cropped_img, (100, 100))
        pose_input = pose_input.astype('float32') / 255.0
        pose_input = np.transpose(pose_input, (2, 0, 1))
        pose_input = np.expand_dims(pose_input, axis=0)

        # 执行推理
        result = exe.run(inference_program, feed={feed_target_names[0]: pose_input}, fetch_list=self.fetch_targets)
        return result[0]  # 假设第一个结果为姿态关键点

    def visualize_pose(self, image, pose_result, x_min, y_min, x_max, y_max, pose_label):
        """
        在原图像上绘制姿态检测结果并打上标签
        """
        # 将 OpenCV 图像转换为 PIL 图像
        im_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(im_pil)

        # 确保标签放在锚框的右下角
        label_pos = (int(x_max), int(y_max))
        print(f"标签位置: {label_pos}")  # 输出调试信息

        # 获取文本大小
        text = pose_label
        text_bbox = draw.textbbox((0, 0), text)  # 使用 textbbox 获取文字的边框
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        label_y_min = max(0, label_pos[1] - text_height)  # 防止标签绘制在图像外部
        label_y_max = label_pos[1]  # 确保 y 坐标的正确顺序

        # zhanli，绿色背景框，tangwo，黄色背景框
        if pose_label == 'zhanli':
            color = 'green'
        elif pose_label == 'tangwo':
            color = 'yellow'

        # 绘制背景的标签框
        draw.rectangle(
            [(label_pos[0], label_y_min), (label_pos[0] + text_width, label_y_max)],
            fill=color
        )

        # 在背景框内绘制白色的标签文字
        draw.text((label_pos[0], label_y_min), text, fill=(255, 255, 255))

        # 在边界框的中心点绘制圆点
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        radius = 5  # 圆点的半径

        circle_color = 'green'
        # zhanli，绿色圆点，tangwo，黄色圆点
        if pose_label == 'zhanli':
            circle_color = 'green'
        elif pose_label == 'tangwo':
            circle_color = 'yellow'

        draw.ellipse(
            [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
            fill=circle_color
        )

        # 将 PIL 图像转换回 OpenCV 格式
        image = np.array(im_pil)

        return image

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        assert isinstance(np_boxes_num, np.ndarray), \
            '`np_boxes_num` should be a `numpy.ndarray`'

        result = {k: v for k, v in result.items() if v is not None}
        return result

    def filter_box(self, result, threshold):
        np_boxes_num = result['boxes_num']
        boxes = result['boxes']
        start_idx = 0
        filter_boxes = []
        filter_num = []
        for i in range(len(np_boxes_num)):
            boxes_num = np_boxes_num[i]
            boxes_i = boxes[start_idx:start_idx + boxes_num, :]
            idx = boxes_i[:, 1] > threshold
            filter_boxes_i = boxes_i[idx, :]
            filter_boxes.append(filter_boxes_i)
            filter_num.append(filter_boxes_i.shape[0])
            start_idx += boxes_num
        boxes = np.concatenate(filter_boxes)
        filter_num = np.array(filter_num)
        filter_res = {'boxes': boxes, 'boxes_num': filter_num}
        return filter_res

    def predict(self, repeats=1, run_benchmark=False):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None

        if run_benchmark:
            for i in range(repeats):
                self.predictor.run()
                paddle.device.cuda.synchronize()
            result = dict(
                boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
            return result

        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if len(output_names) == 1:
                # some exported model can not get tensor 'bbox_num'
                np_boxes_num = np.array([len(np_boxes)])
            else:
                boxes_num = self.predictor.get_output_handle(output_names[1])
                np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ['masks', 'segm']:
                results[k] = np.concatenate(v)
        return results

    def get_timer(self):
        return self.det_times

    def predict_image_slice(self,
                            img_list,
                            slice_size=[640, 640],
                            overlap_ratio=[0.25, 0.25],
                            combine_method='nms',
                            match_threshold=0.6,
                            match_metric='ios',
                            run_benchmark=False,
                            repeats=1,
                            visual=True,
                            save_results=False):
        # slice infer only support bs=1
        results = []
        try:
            import sahi
            from sahi.slicing import slice_image
        except Exception as e:
            print(
                'sahi not found, plaese install sahi. '
                'for example: `pip install sahi`, see https://github.com/obss/sahi.'
            )
            raise e
        num_classes = len(self.pred_config.labels)
        for i in range(len(img_list)):
            ori_image = img_list[i]
            slice_image_result = sahi.slicing.slice_image(
                image=ori_image,
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1])
            sub_img_num = len(slice_image_result)
            merged_bboxs = []
            print('slice to {} sub_samples.', sub_img_num)

            batch_image_list = [
                slice_image_result.images[_ind] for _ind in range(sub_img_num)
            ]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=50, run_benchmark=True)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats, run_benchmark=True)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += 1

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += 1

            st, ed = 0, result['boxes_num'][0]  # start_index, end_index
            for _ind in range(sub_img_num):
                boxes_num = result['boxes_num'][_ind]
                ed = st + boxes_num
                shift_amount = slice_image_result.starting_pixels[_ind]
                result['boxes'][st:ed][:, 2:4] = result['boxes'][
                                                 st:ed][:, 2:4] + shift_amount
                result['boxes'][st:ed][:, 4:6] = result['boxes'][
                                                 st:ed][:, 4:6] + shift_amount
                merged_bboxs.append(result['boxes'][st:ed])
                st = ed

            merged_results = {'boxes': []}
            if combine_method == 'nms':
                final_boxes = multiclass_nms(
                    np.concatenate(merged_bboxs), num_classes, match_threshold,
                    match_metric)
                merged_results['boxes'] = np.concatenate(final_boxes)
            elif combine_method == 'concat':
                merged_results['boxes'] = np.concatenate(merged_bboxs)
            else:
                raise ValueError(
                    "Now only support 'nms' or 'concat' to fuse detection results."
                )
            merged_results['boxes_num'] = np.array(
                [len(merged_results['boxes'])], dtype=np.int32)

            if visual:
                visualize(
                    [ori_image],  # should be list
                    merged_results,
                    self.pred_config.labels,
                    output_dir=self.output_dir,
                    threshold=self.threshold)

            results.append(merged_results)
            print('Test iter {}'.format(i))

        results = self.merge_batch_result(results)
        if save_results:
            Path(self.output_dir).mkdir(exist_ok=True)
            self.save_coco_results(
                img_list, results, use_coco_category=FLAGS.use_coco_category)
        return results

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True,
                      save_results=False):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=50, run_benchmark=True)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats, run_benchmark=True)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                if visual:
                    visualize(
                        batch_image_list,
                        result,
                        self.pred_config.labels,
                        output_dir=self.output_dir,
                        threshold=self.threshold)
            results.append(result)
            print('Test iter {}'.format(i))
        results = self.merge_batch_result(results)
        if save_results:
            Path(self.output_dir).mkdir(exist_ok=True)
            self.save_coco_results(
                image_list, results, use_coco_category=FLAGS.use_coco_category)
        return results

    def predict_video(self, video_file, camera_id=-1, stream_url=None):
        video_out_name = 'output.mp4'
        print(f'stream_url为:{stream_url}')
        if stream_url:
            capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        elif camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_out_name = os.path.split(video_file)[-1]

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        delay = int(1000 / fps)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"fps: {fps}, frame_count: {frame_count}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        index = 1

        last_positions = {}  # 记录上一帧中所有猪的中心点

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            print(f"detect frame: {index}")
            index += 1
            results = self.predict_image([frame[:, :, ::-1]], visual=False)

            # 获取当前时间
            current_time = datetime.now().time()
            is_feeding_time = is_within_feeding_window(current_time, feeding_times, time_window)

            current_positions = {}  # 用于记录当前帧每只猪的位置和ID

            for bbox in results['boxes']:
                class_id, score, x_min, y_min, x_max, y_max = bbox[:6]
                if score > self.threshold:
                    if stream_url == None or is_feeding_time == False:
                        cropped_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                        pose_results = self.pose_predict(self.pose_predictor_exe, self.pose_inference_program,
                                                         self.feed_target_names, cropped_img)
                        print(f'标签:{self.label_list}')
                        pose_label = self.label_list[np.argmax(pose_results)]  # 获取姿态标签
                        print(pose_label)
                        # 更新帧，绘制姿态结果
                        frame = self.visualize_pose(frame, pose_results, x_min, y_min, x_max, y_max, pose_label)
                    else:
                        # 计算边界框的中心点
                        current_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                        pig_id = assign_pig_id(current_center, last_positions)  # 基于边界框生成一个唯一的猪ID
                        current_positions[pig_id] = current_center

                        # 根据姿态判定，此时tangwo的猪被视为食欲不振的猪
                        cropped_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                        pose_results = self.pose_predict(self.pose_predictor_exe, self.pose_inference_program,
                                                         self.feed_target_names, cropped_img)
                        pose = self.label_list[np.argmax(pose_results)]  # 获取姿态标签
                        if pose == 'tangwo':
                            pose_label = 'tangwo'
                            frame = self.visualize_pose(frame, [], x_min, y_min, x_max, y_max, pose_label)

                        # 姿态和是否静止双重判定，确保姿态没有被正确识别的情况下也能识别出食欲不振的猪
                        if pig_id in pig_position_tracker:
                            last_position, last_time = pig_position_tracker[pig_id]

                            # 计算当前位置和上一次位置的中心点距离
                            distance_moved = np.linalg.norm(np.array(current_center) - np.array(last_position))

                            # 检查这只猪是否移动
                            if distance_moved < 10:  # 如果距离小于阈值，判定为静止
                                time_stationary = datetime.now() - last_time
                                if is_feeding_time and time_stationary.total_seconds() > movement_threshold_seconds:
                                    # 如果在投喂时间内长时间不动，标记为不健康的猪
                                    pose_label = "unhealthy pig"
                                    frame = self.visualize_pose(frame, [], x_min, y_min, x_max, y_max, pose_label)
                            else:
                                # 如果位置变化大于阈值，更新位置和时间
                                pig_position_tracker[pig_id] = (current_center, datetime.now())
                        else:
                            # 发现新猪，存储它的位置和检测时间
                            pig_position_tracker[pig_id] = (current_center, datetime.now())

            im = visualize_box_mask(frame, results, self.pred_config.labels, threshold=self.threshold)
            im = np.array(im)
            writer.write(im)

            if camera_id != -1:
                cv2.imshow('Detection and Pose', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif stream_url:
                print("显示推流")
                cv2.imshow('Detection and Pose', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        writer.release()

    def save_coco_results(self, image_list, results, use_coco_category=False):
        bbox_results = []
        mask_results = []
        idx = 0
        print("Start saving coco json files...")
        for i, box_num in enumerate(results['boxes_num']):
            file_name = os.path.split(image_list[i])[-1]
            if use_coco_category:
                img_id = int(os.path.splitext(file_name)[0])
            else:
                img_id = i

            if 'boxes' in results:
                boxes = results['boxes'][idx:idx + box_num].tolist()
                bbox_results.extend([{
                    'image_id': img_id,
                    'category_id': coco_clsid2catid[int(box[0])] \
                        if use_coco_category else int(box[0]),
                    'file_name': file_name,
                    'bbox': [box[2], box[3], box[4] - box[2],
                             box[5] - box[3]],  # xyxy -> xywh
                    'score': box[1]} for box in boxes])

            if 'masks' in results:
                import pycocotools.mask as mask_util

                boxes = results['boxes'][idx:idx + box_num].tolist()
                masks = results['masks'][i][:box_num].astype(np.uint8)
                seg_res = []
                for box, mask in zip(boxes, masks):
                    rle = mask_util.encode(
                        np.array(
                            mask[:, :, None], dtype=np.uint8, order="F"))[0]
                    if 'counts' in rle:
                        rle['counts'] = rle['counts'].decode("utf8")
                    seg_res.append({
                        'image_id': img_id,
                        'category_id': coco_clsid2catid[int(box[0])] \
                            if use_coco_category else int(box[0]),
                        'file_name': file_name,
                        'segmentation': rle,
                        'score': box[1]})
                mask_results.extend(seg_res)

            idx += box_num

        if bbox_results:
            bbox_file = os.path.join(self.output_dir, "bbox.json")
            with open(bbox_file, 'w') as f:
                json.dump(bbox_results, f)
            print(f"The bbox result is saved to {bbox_file}")
        if mask_results:
            mask_file = os.path.join(self.output_dir, "mask.json")
            with open(mask_file, 'w') as f:
                json.dump(mask_results, f)
            print(f"The mask result is saved to {mask_file}")


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0],)).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'],)).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'],)).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'],)).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'],)).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print(
                'The RCNN export model is used for ONNX and it only supports batch_size = 1'
            )
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
                                                                      'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir,
                   arch,
                   run_mode='paddle',
                   batch_size=1,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_mkldnn_bfloat16=False,
                   delete_shuffle_pass=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT.
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
            .format(run_mode, device))
    infer_model = os.path.join(model_dir, 'model.pdmodel')
    infer_params = os.path.join(model_dir, 'model.pdiparams')
    if not os.path.exists(infer_model):
        infer_model = os.path.join(model_dir, 'inference.pdmodel')
        infer_params = os.path.join(model_dir, 'inference.pdiparams')
        if not os.path.exists(infer_model):
            raise ValueError(
                "Cannot find any inference model in dir: {},".format(model_dir))
    config = Config(infer_model, infer_params)
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    elif device == 'NPU':
        if hasattr(config, "lite_engine_enabled"):
            if config.lite_engine_enabled():
                config.enable_lite_engine()
        config.enable_custom_device('npu')
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=(1 << 25) * batch_size,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)
        if FLAGS.collect_trt_shape_info:
            config.collect_shape_range_info(FLAGS.tuned_trt_shape_file)
        elif os.path.exists(FLAGS.tuned_trt_shape_file):
            print(f'Use dynamic shape file: '
                  f'{FLAGS.tuned_trt_shape_file} for TRT...')
            config.enable_tuned_tensorrt_dynamic_shape(
                FLAGS.tuned_trt_shape_file, True)

        if use_dynamic_shape:
            min_input_shape = {
                'image': [batch_size, 3, trt_min_shape, trt_min_shape],
                'scale_factor': [batch_size, 2]
            }
            max_input_shape = {
                'image': [batch_size, 3, trt_max_shape, trt_max_shape],
                'scale_factor': [batch_size, 2]
            }
            opt_input_shape = {
                'image': [batch_size, 3, trt_opt_shape, trt_opt_shape],
                'scale_factor': [batch_size, 2]
            }
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--image_file or --image_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
        "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
        "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images


def visualize(image_list, result, labels, output_dir='output/', threshold=0.5):
    # visualize the predict result
    start_idx = 0
    for idx, image_file in enumerate(image_list):
        im_bboxes_num = result['boxes_num'][idx]
        im_results = {}
        if 'boxes' in result:
            im_results['boxes'] = result['boxes'][start_idx:start_idx +
                                                            im_bboxes_num, :]
        if 'masks' in result:
            im_results['masks'] = result['masks'][start_idx:start_idx +
                                                            im_bboxes_num, :]
        if 'segm' in result:
            im_results['segm'] = result['segm'][start_idx:start_idx +
                                                          im_bboxes_num, :]
        if 'label' in result:
            im_results['label'] = result['label'][start_idx:start_idx +
                                                            im_bboxes_num]
        if 'score' in result:
            im_results['score'] = result['score'][start_idx:start_idx +
                                                            im_bboxes_num]

        start_idx += im_bboxes_num
        im = visualize_box_mask(
            image_file, im_results, labels, threshold=threshold)
        img_name = os.path.split(image_file)[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, img_name)
        im.save(out_path, quality=95)
        print("save result to: " + out_path)


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


def main():
    parser = argsparser()
    FLAGS = parser.parse_args()

    print(f'FLAGS.stream_url:{FLAGS.stream_url}')

    # 创建检测器实例并执行预测
    detector = Detector(
        model_dir=FLAGS.model_dir,
        pose_model_dir="VGG16_adagrad",  # 姿态检测模型路径
        label_file="zishi/readme.json",  # 姿态标签文件路径
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        enable_mkldnn_bfloat16=FLAGS.enable_mkldnn_bfloat16,
        threshold=FLAGS.threshold,
        output_dir=FLAGS.output_dir
    )

    if FLAGS.video_file:
        detector.predict_video(video_file=FLAGS.video_file)
    elif FLAGS.camera_id != -1:
        detector.predict_video(video_file=None, camera_id=FLAGS.camera_id)
    elif FLAGS.stream_url:
        detector.predict_video(video_file=None, camera_id=-1, stream_url=FLAGS.stream_url)
    else:
        print("Please provide video input!")


if __name__ == '__main__':
    paddle.enable_static()
    main()
