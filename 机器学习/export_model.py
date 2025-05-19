import argparse
import os
import torch
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.resolve()

def parse_args():
    parser = argparse.ArgumentParser(description='导出水果识别模型')
    parser.add_argument('--weights', type=str, default=ROOT / 'models/best.pt', 
                        help='训练好的权重文件路径')
    parser.add_argument('--img-size', type=int, default=320, 
                        help='导出模型的输入图像尺寸')
    parser.add_argument('--format', type=str, default='onnx', 
                        choices=['torchscript', 'onnx', 'tflite', 'openvino', 'paddle', 'ncnn'],
                        help='导出格式')
    parser.add_argument('--device', default='', 
                        help='cuda设备，例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--include', nargs='+', default=['onnx'],
                        help='可选的导出格式列表')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"开始导出模型: {args.weights}")
    
    # 检查权重文件是否存在
    weights = Path(args.weights)
    if not weights.exists():
        print(f"错误: 权重文件 {weights} 不存在")
        return
    
    # 创建输出目录
    output_dir = ROOT / 'models/exported'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 构建导出命令
    export_script = ROOT / "yolov5/export.py"
    export_cmd = f"python {export_script} --weights {args.weights} --img {args.img_size} --include {args.format} --device {args.device}"
    
    print(f"执行命令: {export_cmd}")
    os.system(export_cmd)
    
    # 显示导出结果
    export_formats = {
        'torchscript': '.pt',
        'onnx': '.onnx',
        'tflite': '.tflite',
        'openvino': '_openvino_model',
        'paddle': '_paddle_model',
        'ncnn': '_ncnn_model'
    }
    
    # 检查是否成功导出
    weight_name = weights.stem
    if args.format in export_formats:
        ext = export_formats[args.format]
        if args.format in ['openvino', 'paddle', 'ncnn']:
            exported_file = ROOT / f"{weight_name}{ext}"
            if exported_file.exists():
                print(f"成功导出模型: {exported_file}")
            else:
                print(f"警告: 未找到导出的模型文件 {exported_file}")
        else:
            exported_file = ROOT / f"{weight_name}{ext}"
            if exported_file.exists():
                print(f"成功导出模型: {exported_file}")
                print(f"模型大小: {exported_file.stat().st_size / (1024 * 1024):.2f} MB")
            else:
                print(f"警告: 未找到导出的模型文件 {exported_file}")

if __name__ == "__main__":
    main()
