import os
import sys
import cv2
import numpy as np
import serial
import time
from PIL import Image
import torch

# 确保正确路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

# 导入YOLOv5模块
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# 配置参数
weights = 'yolov5s.pt'  # 模型权重
img_size = 640  # 推理尺寸
conf_thres = 0.25  # 置信度阈值
iou_thres = 0.45  # NMS IOU阈值
device = ''  # 设备选择

def get_serial_data(port, baudrate=115200, timeout=1):
    """
    从串口获取图像数据
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"成功打开串口: {port}")
        
        # 等待数据
        print("等待摄像头数据...")
        
        # 读取图像宽度和高度
        width_bytes = ser.read(2)
        height_bytes = ser.read(2)
        
        if len(width_bytes) < 2 or len(height_bytes) < 2:
            print("无法读取图像尺寸数据")
            return None
            
        width = int.from_bytes(width_bytes, byteorder='little')
        height = int.from_bytes(height_bytes, byteorder='little')
        print(f"图像尺寸: {width}x{height}")
        
        # 计算期望的数据大小
        expected_size = width * height * 3  # RGB每像素3字节
        
        # 读取图像数据
        image_data = bytearray()
        remaining = expected_size
        
        while remaining > 0:
            chunk = ser.read(min(1024, remaining))
            if not chunk:
                break
            image_data.extend(chunk)
            remaining -= len(chunk)
            
            # 显示进度
            percent = ((expected_size - remaining) / expected_size) * 100
            print(f"接收数据: {percent:.1f}%", end='\r')
            
        print("\n数据接收完毕")
        
        if len(image_data) != expected_size:
            print(f"数据接收不完整: {len(image_data)}/{expected_size}")
            return None
            
        # 转换为OpenCV图像格式
        nparr = np.frombuffer(image_data, np.uint8)
        img = nparr.reshape((height, width, 3))
        
        return img
        
    except Exception as e:
        print(f"串口通信错误: {e}")
        return None
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

def detect_fruits(image):
    """
    使用YOLOv5模型检测水果
    """
    # 初始化
    device = select_device(device)
    model = attempt_load(weights, device=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步长
    imgsz = check_img_size(img_size, s=stride)  # 检查图像大小
    
    # 转换图像
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    # 预处理
    img = np.array(img)
    img = img.transpose(2, 0, 1)  # HWC到CHW
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0 - 255 转 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        pred = model(img, augment=False)[0]
    
    # 应用NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # 处理检测结果
    results = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in det:
                results.append({
                    'box': xyxy,
                    'confidence': conf.item(),
                    'class': int(cls.item()),
                    'class_name': model.names[int(cls.item())]
                })
    
    return results

def draw_results(image, results):
    """
    在图像上绘制检测结果
    """
    for result in results:
        box = [int(x.item()) for x in result['box']]
        label = f"{result['class_name']} {result['confidence']:.2f}"
        plot_one_box(box, image, label=label, color=(0, 255, 0))
    
    return image

def main():
    # 命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=str, required=True, help='串口号，如COM3')
    args = parser.parse_args()
    
    # 获取图像数据
    image = get_serial_data(args.port)
    
    if image is None:
        print("未能获取有效图像")
        return
        
    # 检测水果
    results = detect_fruits(image)
    
    # 绘制结果
    image_with_results = draw_results(image.copy(), results)
    
    # 显示结果
    cv2.imshow("Fruit Detection", image_with_results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("检测到的水果:")
    for result in results:
        print(f"{result['class_name']}: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()
