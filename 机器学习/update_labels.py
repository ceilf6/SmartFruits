import os
import argparse
import yaml
import shutil
from pathlib import Path
from PIL import Image
import random

# 项目根目录
ROOT = Path(__file__).parent.resolve()

def parse_args():
    parser = argparse.ArgumentParser(description='更新水果数据集标签')
    parser.add_argument('--data', type=str, default=ROOT / 'dataset/fruits.yaml', 
                       help='数据集配置文件')
    parser.add_argument('--add-class', type=str, default=None,
                       help='要添加的新水果类别名称')
    return parser.parse_args()

def update_yaml(yaml_file, new_class=None):
    """更新YAML文件中的类别列表"""
    if not yaml_file.exists():
        print(f"错误: 文件 {yaml_file} 不存在")
        return False
        
    # 读取现有配置
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    if not data:
        print(f"错误: {yaml_file} 内容为空或格式错误")
        return False
        
    if 'names' not in data:
        print(f"错误: {yaml_file} 中没有找到names字段")
        return False
    
    # 如果指定了新类别
    if new_class:
        if new_class in data['names']:
            print(f"类别 '{new_class}' 已存在，索引为 {data['names'].index(new_class)}")
        else:
            # 添加新类别
            data['names'].append(new_class)
            print(f"已添加新类别 '{new_class}', 索引为 {len(data['names']) - 1}")

            # 备份原文件
            backup_file = yaml_file.parent / f"{yaml_file.stem}_backup.yaml"
            shutil.copy2(yaml_file, backup_file)
            print(f"已创建备份文件: {backup_file}")
            
            # 保存更新后的文件
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True)
                
    # 打印当前类别
    print(f"当前配置包含 {len(data['names'])} 个水果类别:")
    for idx, name in enumerate(data['names']):
        print(f"  {idx}: {name}")
            
    return True

def main():
    args = parse_args()
    print(f"更新水果数据集标签...")
    
    # 更新YAML配置文件
    if not update_yaml(Path(args.data), args.add_class):
        return
        
    # 检查数据集目录结构
    data_dir = Path(args.data).parent
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    
    # 检查各个目录是否存在
    print("\n检查数据集目录结构:")
    for d in [images_dir, labels_dir]:
        for split in ['train', 'val']:
            check_dir = d / split
            if check_dir.exists():
                files = list(check_dir.glob('*.*'))
                print(f"  {check_dir}: {len(files)} 个文件")
            else:
                print(f"  {check_dir}: 目录不存在")
                check_dir.mkdir(parents=True, exist_ok=True)
                print(f"  已创建目录: {check_dir}")

if __name__ == "__main__":
    main()
