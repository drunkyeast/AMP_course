#!/usr/bin/env python3
"""
测试终端键盘控制功能
"""
import sys
import select
import tty
import termios
from threading import Thread
import time

# 全局变量
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 3.5, 1.0, 1.0
vel_increment = 0.1

def getch():
    """获取单个字符输入，不需要回车"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # 检测方向键（ESC序列）
        if ch == '\x1b':  # ESC
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def print_status():
    """打印当前状态"""
    print(f"\r速度状态: X={x_vel_cmd:+.2f} Y={y_vel_cmd:+.2f} Yaw={yaw_vel_cmd:+.2f} | 使用方向键控制，q退出", end='', flush=True)

def handle_terminal_keyboard():
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    
    print("终端键盘控制测试已启动:")
    print("↑ - 增加前进速度 (x+)")
    print("↓ - 减少前进速度 (x-)")
    print("← - 增加左移速度 (y+)")
    print("→ - 增加右移速度 (y-)")
    print("a/d - 控制转向 (yaw)")
    print("空格 - 停止所有运动")
    print("q - 退出键盘控制")
    print("-" * 50)
    
    print_status()
    
    while True:
        try:
            # 非阻塞检查是否有输入
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = getch()
                
                if key == 'q':
                    print("\n键盘控制已退出")
                    break
                elif key == '\x1b[A':  # 上箭头
                    x_vel_cmd = min(x_vel_cmd + vel_increment, x_vel_max)
                    print_status()
                elif key == '\x1b[B':  # 下箭头
                    x_vel_cmd = max(x_vel_cmd - vel_increment, -x_vel_max)
                    print_status()
                elif key == '\x1b[D':  # 左箭头
                    y_vel_cmd = min(y_vel_cmd + vel_increment, y_vel_max)
                    print_status()
                elif key == '\x1b[C':  # 右箭头
                    y_vel_cmd = max(y_vel_cmd - vel_increment, -y_vel_max)
                    print_status()
                elif key == 'a':
                    yaw_vel_cmd = min(yaw_vel_cmd + vel_increment, yaw_vel_max)
                    print_status()
                elif key == 'd':
                    yaw_vel_cmd = max(yaw_vel_cmd - vel_increment, -yaw_vel_max)
                    print_status()
                elif key == ' ':  # 空格
                    x_vel_cmd = 0.0
                    y_vel_cmd = 0.0
                    yaw_vel_cmd = 0.0
                    print_status()
        except KeyboardInterrupt:
            print("\n键盘控制已中断")
            break
        except Exception as e:
            print(f"\n键盘输入错误: {e}")
            break

def main():
    # 启动键盘控制线程
    keyboard_thread = Thread(target=handle_terminal_keyboard)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    # 主循环模拟机器人控制
    try:
        time.sleep(1)  # 等待键盘线程启动
        while True:
            time.sleep(0.5)  # 减少打印频率
    except KeyboardInterrupt:
        print("\n程序退出")

if __name__ == "__main__":
    main()
