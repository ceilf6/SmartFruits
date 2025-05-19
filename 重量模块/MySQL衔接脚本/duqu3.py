import serial
import time
import mysql.connector
from datetime import datetime
import sys

# 串口配置
SERIAL_PORT = 'COM3'  # 根据实际情况修改
BAUD_RATE = 9600

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'weight_data'
}

# 水果类型映射（由串口发送的代码映射到实际类型）
FRUIT_TYPES = {
    '1': '苹果',
    '2': '香蕉',
    '3': '橙子',
    '4': '草莓',
    '5': '猕猴桃',
    '6': '柠檬',
    '7': '葡萄',
    '8': '西瓜',
    '9': '梨',
    '10': '桃子',
    # 可以添加更多水果类型
}

def setup_database():
    """
    设置数据库，创建表（如果不存在）
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 创建水果重量记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fruit_weights (
            id INT AUTO_INCREMENT PRIMARY KEY,
            fruit_type VARCHAR(50) NOT NULL,
            weight FLOAT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("数据库设置完成")
        return True
        
    except mysql.connector.Error as e:
        print(f"数据库设置失败: {e}")
        return False

def save_to_database(fruit_type, weight):
    """
    将重量数据保存到数据库
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 插入记录
        cursor.execute(
            "INSERT INTO fruit_weights (fruit_type, weight) VALUES (%s, %s)",
            (fruit_type, weight)
        )
        
        conn.commit()
        record_id = cursor.lastrowid
        cursor.close()
        conn.close()
        
        print(f"数据成功保存到数据库，记录ID: {record_id}")
        return record_id
        
    except mysql.connector.Error as e:
        print(f"数据库保存失败: {e}")
        return None

def read_serial_data():
    """
    从串口读取数据
    数据格式：fruit_code,weight
    例如：1,156.78 代表一个苹果，重156.78克
    """
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"串口 {SERIAL_PORT} 已打开")
        
        # 等待串口初始化
        time.sleep(2)
        
        # 读取并清空缓冲区
        ser.flushInput()
        
        print("等待重量数据...")
        
        while True:
            # 读取一行数据
            line = ser.readline().decode('utf-8').strip()
            
            # 如果有数据
            if line:
                print(f"接收到数据: {line}")
                
                try:
                    # 解析数据
                    parts = line.split(',')
                    if len(parts) == 2:
                        fruit_code = parts[0]
                        weight = float(parts[1])
                        
                        # 获取水果类型
                        fruit_type = FRUIT_TYPES.get(fruit_code, f"未知类型({fruit_code})")
                        
                        print(f"解析结果: {fruit_type}, {weight}g")
                        
                        # 保存到数据库
                        record_id = save_to_database(fruit_type, weight)
                        
                        if record_id:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                                  f"记录 #{record_id}: {fruit_type}, {weight}g")
                    else:
                        print(f"数据格式错误: {line}")
                except Exception as e:
                    print(f"数据处理错误: {e}")
            
            # 短暂延时
            time.sleep(0.1)
            
    except serial.SerialException as e:
        print(f"串口错误: {e}")
        return False
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("串口已关闭")
    
    return True

def main():
    """
    主函数
    """
    print("水果重量测量系统 - 数据读取程序")
    print("=" * 50)
    
    # 设置数据库
    if not setup_database():
        print("无法继续，程序退出")
        sys.exit(1)
    
    # 读取串口数据
    read_serial_data()

if __name__ == "__main__":
    main()
