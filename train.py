import os
from ultralytics import YOLO

if __name__ == '__main__':
    yaml_yolov8s = 'ultralytics/cfg/models/v8/cls_self/yolov8s-cls.yaml'
    yaml_yolov8_SE = 'ultralytics/cfg/models/v8/cls_self/yolov8s-cls-atten-SE.yaml'
    model_yaml = yaml_yolov8s
    # 模型加载
    model = YOLO(model_yaml)
    data_path = r'C:\Users\he\Desktop\traindata\traindata'
    name = os.path.basename(model_yaml).split('.')[0]
    model.train(data=data_path,             # 数据集路径
                imgsz=300,                  # 训练图片大小
                epochs=200,                 # 训练的轮次
                batch=2,                    # 训练batch
                workers=1,                  # 加载数据线程数
                device='0',                 # 使用显卡
                optimizer='SGD',            # 优化器
                project='runs/train',       # 模型保存路径
                name=name,                  # 模型保存命名
                )
