import os
import argparse
import yaml
import torch
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.resolve()

def parse_args():
    parser = argparse.ArgumentParser(description='继续训练水果识别模型')
    parser.add_argument('--weights', type=str, default=ROOT / 'models/last.pt', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default=ROOT / 'yolov5/models/yolov5s.yaml', help='模型配置文件')
    parser.add_argument('--data', type=str, default=ROOT / 'dataset/fruits.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='图像大小')
    parser.add_argument('--device', default='', help='cuda设备，例如 0 或 0,1,2,3 或 cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"继续训练水果识别模型...")
    print(f"加载权重: {args.weights}")
    print(f"数据集配置: {args.data}")
    
    # 检查权重文件是否存在
    weights = Path(args.weights)
    if not weights.exists():
        print(f"错误: 权重文件 {weights} 不存在")
        return
        
    # 检查数据配置文件是否存在
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"错误: 数据配置文件 {data_yaml} 不存在")
        return
        
    # 加载数据配置
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
        print(f"数据集包含 {len(data_dict['names'])} 种水果类别: {data_dict['names']}")

    # 构建训练命令
    train_script = Path(ROOT) / "yolov5/train.py"
    train_command = f"python {train_script} --weights {args.weights} --data {args.data} --epochs {args.epochs} --batch-size {args.batch_size} --img {args.img_size} --device {args.device}"
    
    print(f"执行命令: {train_command}")
    os.system(train_command)

if __name__ == "__main__":
    main()
