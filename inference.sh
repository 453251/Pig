
# 对视频进行推理
python PaddleYOLO/deploy/python/infer_pose.py --model_dir=output_model_new3/yolov8_s_500e_coco --video_file pig.mp4 --device=GPU --output_dir output_video_pose --save_images True