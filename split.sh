python /home/aistudio/PaddleYOLO/tools/x2coco.py \
                --dataset_type labelme \
                --json_input_dir /home/aistudio/train_json/ \
                --image_input_dir /home/aistudio/train_img/ \
                --output_dir /home/aistudio/data/coco/ \
                --train_proportion 0.8 \
                --val_proportion 0.2 \
                --test_proportion 0.0