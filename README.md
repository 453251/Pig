本项目为使用Paddle框架实现的生猪防疫系统。主要通过姿态识别和离群行为判定病猪。使用yolov8实现目标检测，训练VGG网络作为简单的二元分类器用于简单姿态识别，将目标检测得到的锚框作为VGG网络的输入进行姿态检测
同时设计了离群判定算法，主要根据最短距离和锚框的交并比进行综合判定
此外，还使用OBS Studio循环RTSP推流一段猪场监控视频，模拟真实监控场景
由于PaddleYOLO套件不能整个直接传，因此只列出我重写的几个重要文件，分别是utils增加监控推流参数，visualize_pose增加离群框绘制，infer_pose增加关键点检测模块和监控推流模块，x2coco是重写的把科大讯飞的生猪数据集转换成coco标注格式的脚本
output_video_pose是导出的检测视频的文件夹
