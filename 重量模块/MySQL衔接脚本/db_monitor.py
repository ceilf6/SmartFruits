import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mysql.connector
from datetime import datetime, timedelta
import numpy as np

class WeightDataMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("水果重量数据监控系统")
        self.root.geometry("1200x700")
        
        # 数据库连接设置
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '123456',
            'database': 'weight_data'
        }
        
        # 创建界面
        self.create_ui()
        
        # 初始加载数据
        self.load_data()
        
    def create_ui(self):
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部控制栏
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # 时间范围选择
        ttk.Label(control_frame, text="数据范围:").pack(side=tk.LEFT, padx=5)
        
        self.time_range = tk.StringVar(value="today")
        ttk.Radiobutton(control_frame, text="今天", variable=self.time_range, 
                       value="today", command=self.load_data).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="昨天", variable=self.time_range,
                       value="yesterday", command=self.load_data).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="本周", variable=self.time_range,
                       value="week", command=self.load_data).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="本月", variable=self.time_range,
                       value="month", command=self.load_data).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="所有数据", variable=self.time_range,
                       value="all", command=self.load_data).pack(side=tk.LEFT)
        
        # 刷新按钮
        ttk.Button(control_frame, text="刷新数据", command=self.load_data).pack(side=tk.RIGHT, padx=10)
        
        # 创建左右分栏
        paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 左侧数据表格
        table_frame = ttk.LabelFrame(paned_window, text="重量数据记录")
        paned_window.add(table_frame, weight=1)
        
        # 创建表格
        self.tree = ttk.Treeview(table_frame, columns=("id", "fruit_type", "weight", "time"), show="headings")
        self.tree.heading("id", text="ID")
        self.tree.heading("fruit_type", text="水果类型")
        self.tree.heading("weight", text="重量(g)")
        self.tree.heading("time", text="测量时间")
        
        self.tree.column("id", width=50)
        self.tree.column("fruit_type", width=100)
        self.tree.column("weight", width=100)
        self.tree.column("time", width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 右侧图表区域
        chart_frame = ttk.LabelFrame(paned_window, text="数据可视化")
        paned_window.add(chart_frame, weight=2)
        
        # 创建选项卡
        self.tab_control = ttk.Notebook(chart_frame)
        
        # 水果类型分布选项卡
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text="水果类型分布")
        
        # 重量分布选项卡
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab2, text="重量分布")
        
        # 时间趋势选项卡
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab3, text="时间趋势")
        
        self.tab_control.pack(expand=1, fill=tk.BOTH)
        
        # 状态栏
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_data(self):
        # 根据选择的时间范围构建SQL查询
        time_filter = self.get_time_filter()
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            query = f"SELECT id, fruit_type, weight, timestamp FROM fruit_weights {time_filter} ORDER BY timestamp DESC"
            cursor.execute(query)
            records = cursor.fetchall()
            
            # 清空表格
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            # 填充数据
            for row in records:
                self.tree.insert("", tk.END, values=row)
                
            # 更新状态栏
            self.status_var.set(f"已加载 {len(records)} 条记录")
            
            # 将数据转换为DataFrame
            self.df = pd.DataFrame(records, columns=['id', 'fruit_type', 'weight', 'timestamp'])
            
            # 更新图表
            self.update_charts()
            
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as e:
            messagebox.showerror("数据库错误", f"无法连接到数据库或查询失败: {e}")
            self.status_var.set("数据加载失败")
    
    def get_time_filter(self):
        """根据选择的时间范围返回SQL WHERE子句"""
        range_value = self.time_range.get()
        
        if range_value == "all":
            return ""
            
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if range_value == "today":
            start_date = today
            end_date = today + timedelta(days=1)
        elif range_value == "yesterday":
            start_date = today - timedelta(days=1)
            end_date = today
        elif range_value == "week":
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(days=7)
        elif range_value == "month":
            start_date = today.replace(day=1)
            if today.month == 12:
                end_date = today.replace(year=today.year+1, month=1, day=1)
            else:
                end_date = today.replace(month=today.month+1, day=1)
        
        return f"WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
        
    def update_charts(self):
        """更新所有图表"""
        if self.df.empty:
            return
            
        # 更新水果类型分布图
        self.update_fruit_type_chart()
        
        # 更新重量分布图
        self.update_weight_distribution_chart()
        
        # 更新时间趋势图
        self.update_time_trend_chart()
    
    def update_fruit_type_chart(self):
        """更新水果类型分布饼图"""
        # 清空当前图表
        for widget in self.tab1.winfo_children():
            widget.destroy()
            
        # 创建新的图表
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 统计水果类型分布
        fruit_counts = self.df['fruit_type'].value_counts()
        
        # 绘制饼图
        ax.pie(fruit_counts, labels=fruit_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # 确保饼图是圆的
        ax.set_title('水果类型分布')
        
        # 嵌入到Tkinter界面
        canvas = FigureCanvasTkAgg(fig, self.tab1)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def update_weight_distribution_chart(self):
        """更新重量分布直方图"""
        # 清空当前图表
        for widget in self.tab2.winfo_children():
            widget.destroy()
            
        # 创建新的图表
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 绘制直方图
        ax.hist(self.df['weight'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('重量 (g)')
        ax.set_ylabel('频率')
        ax.set_title('水果重量分布')
        
        # 添加均值线
        mean_weight = self.df['weight'].mean()
        ax.axvline(mean_weight, color='red', linestyle='--', linewidth=1)
        ax.text(mean_weight*1.05, ax.get_ylim()[1]*0.9, f'均值: {mean_weight:.2f}g')
        
        # 嵌入到Tkinter界面
        canvas = FigureCanvasTkAgg(fig, self.tab2)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def update_time_trend_chart(self):
        """更新时间趋势线图"""
        # 清空当前图表
        for widget in self.tab3.winfo_children():
            widget.destroy()
            
        # 创建新的图表
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # 转换时间戳为datetime对象并排序
        self.df['date'] = pd.to_datetime(self.df['timestamp'])
        daily_data = self.df.groupby(self.df['date'].dt.date).agg({
            'id': 'count',
            'weight': 'mean'
        }).reset_index()
        
        # 创建双坐标轴
        ax2 = ax.twinx()
        
        # 绘制每日测量次数线图
        ax.plot(daily_data['date'], daily_data['id'], '-o', color='blue', label='测量次数')
        ax.set_xlabel('日期')
        ax.set_ylabel('测量次数', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # 绘制每日平均重量线图
        ax2.plot(daily_data['date'], daily_data['weight'], '-s', color='green', label='平均重量')
        ax2.set_ylabel('平均重量 (g)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # 设置x轴日期格式
        fig.autofmt_xdate()
        
        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.set_title('每日测量趋势')
        
        # 嵌入到Tkinter界面
        canvas = FigureCanvasTkAgg(fig, self.tab3)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = WeightDataMonitor(root)
    root.mainloop()
