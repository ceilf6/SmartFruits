import os
import argparse
import yaml
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.resolve()

def parse_args():
    parser = argparse.ArgumentParser(description='训练水果识别模型')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default=ROOT / 'yolov5/models/yolov5s.yaml', help='模型配置文件')
    parser.add_argument('--data', type=str, default=ROOT / 'dataset/fruits.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='图像大小')
    parser.add_argument('--device', default='', help='cuda设备，例如 0 或 0,1,2,3 或 cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"开始训练水果识别模型...")
    print(f"使用权重: {args.weights}")
    print(f"数据集配置: {args.data}")
    
    # 检查数据配置文件是否存在
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"错误: 数据配置文件 {data_yaml} 不存在")
        return
        
    # 加载数据配置
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
        print(f"数据集包含 {len(data_dict['names'])} 种水果类别: {data_dict['names']}")
        
    # 检查数据集路径是否存在
    train_path = Path(data_dict['train'])
    val_path = Path(data_dict['val'])
    
    if not train_path.exists():
        print(f"警告: 训练集路径 {train_path} 不存在")
    else:
        train_images = list(train_path.glob('*.jpg'))
        print(f"训练集包含 {len(train_images)} 张图片")
        
    if not val_path.exists():
        print(f"警告: 验证集路径 {val_path} 不存在")
    else:
        val_images = list(val_path.glob('*.jpg'))
        print(f"验证集包含 {len(val_images)} 张图片")
    
    # 构建训练命令
    train_script = ROOT / "yolov5/train.py"
    train_command = f"python {train_script} --weights {args.weights} --cfg {args.cfg} --data {args.data} --epochs {args.epochs} --batch-size {args.batch_size} --img {args.img_size} --device {args.device}"
    
    print(f"执行命令: {train_command}")
    os.system(train_command)
    
    # 检查训练结果
    runs_dir = ROOT / "yolov5/runs/train"
    if runs_dir.exists():
        exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('exp')]
        if exp_dirs:
            latest_exp = max(exp_dirs, key=os.path.getmtime)
            weights_dir = latest_exp / 'weights'
            if weights_dir.exists():
                best_weights = weights_dir / 'best.pt'
                if best_weights.exists():
                    print(f"训练完成，最佳权重文件: {best_weights}")
                    
                    # 复制权重到项目models目录
                    models_dir = ROOT / 'models'
                    models_dir.mkdir(exist_ok=True)
                    import shutil
                    shutil.copy2(best_weights, models_dir / 'best.pt')
                    print(f"已将最佳权重复制到: {models_dir / 'best.pt'}")
                else:
                    print("训练完成，但未找到最佳权重文件")
            else:
                print("训练完成，但未找到权重目录")
        else:
            print("训练完成，但未找到训练结果目录")
    else:
        print("训练完成，但未找到训练结果父目录")

if __name__ == "__main__":
    main()
