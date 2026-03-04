from ultralytics import YOLO
a=YOLO("yolov8n.pt")
a('D:/BaiduNetdiskDownload/code24_yolov8/ultralytics/assets/bus.jpg',show=True,save=True)
a(r'C:\Users\he\Desktop\popo.jpg',show=True,save=True)