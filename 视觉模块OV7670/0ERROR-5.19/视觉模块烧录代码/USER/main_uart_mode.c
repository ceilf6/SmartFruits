#include "stm32f4xx.h"
#include "delay.h"
#include "lcd.h"
#include "spi.h"
#include "test.h"
#include "GUI.h"
#include "ov7670.h"
#include "sccb.h"
#include "usart.h"
#include "timer.h"
#include "exti.h"
#include "stm32f4xx_gpio.h"

extern u8 ov_sta;  // 在 exit.c 中定义
extern u8 ov_frame;  // 在 time.c 中定义

// 更新LCD显示
void camera_refresh(void) {
    u32 j;
    u16 color;

    if (ov_sta) {  // 有新的一帧图像捕获成功
        LCD_direction(1);
        LCD_SetWindows(0, 0, 319, 239);

        OV7670_RRST = 0;  // 开始复位读指针
        OV7670_RCK_L;
        OV7670_RCK_H;
        OV7670_RCK_L;
        OV7670_RRST = 1;  // 复位读指针结束
        OV7670_RCK_H;

        for (j = 0; j < 76800; j++) {  // 读取数据
            OV7670_RCK_L;
            color = GPIOA->IDR & 0xFF;  // 读数据
            OV7670_RCK_H;
            color <<= 8;

            OV7670_RCK_L;
            color |= GPIOA->IDR & 0xFF;  // 读数据
            OV7670_RCK_H;
            Lcd_WriteData_16Bit(color);
        }

        EXTI_ClearITPendingBit(EXTI_Line8);  // 清除中断标志位
        ov_sta = 0;  // 开始下一次采集
        ov_frame++;
    }
}

// 通过串口发送图像
void camera_refresh_1(void) {
    u32 j;
    u8 data1, data2;

    if (ov_sta) {  // 有新的一帧图像捕获成功
        OV7670_RRST = 0;  // 开始复位读指针
        OV7670_RCK_L;
        OV7670_RCK_H;
        OV7670_RCK_L;
        OV7670_RRST = 1;  // 复位读指针结束
        OV7670_RCK_H;

        printf("%c", 0x01);  // 帧起始标记
        printf("%c", 0xFE);

        for (j = 0; j < 76800; j++) {  // 读取数据
            OV7670_RCK_L;
            data1 = GPIOA->IDR & 0xFF;  // 读数据
            OV7670_RCK_H;

            OV7670_RCK_L;
            data2 = GPIOA->IDR & 0xFF;  // 读数据
            OV7670_RCK_H;

            // 使用printf可能会导致数据发送速度较慢，这里改用直接发送
            while(!(USART1->SR & USART_SR_TXE));
            USART1->DR = data1;
            
            while(!(USART1->SR & USART_SR_TXE));
            USART1->DR = data2;
        }

        printf("%c", 0xFE);  // 帧结束标记
        printf("%c", 0x01);

        EXTI_ClearITPendingBit(EXTI_Line8);  // 清除中断标志位
        ov_sta = 0;  // 开始下一次采集
        ov_frame++;
    }
}

int main(void) {
    // 系统初始化
    SystemInit();  // 配置RCC,配置系统时钟为72MHz
    delay_init(72);  // 延时初始化
    SPI2_Init();  // 配置SPI2接口类型
    LCD_Init();  // 初始化液晶屏
      // 配置串口，使用适中波特率以提高稳定性
    uart_init(460800);  // 使用460800波特率初始化串口

    // 初始化OV7670
    while (OV7670_Init()) {
        LCD_ShowString(100, 20, 16, "ERROR", 1);
        delay_ms(200);
        delay_ms(200);
    }

    LCD_ShowString(200, 200, 16, "ok", 1);
    delay_ms(1500);

    LCD_Clear(BLACK);
    EXTI8_Init();  // 使能帧中断
    OV7670_Effects_Set();  // 设置OV7670效果
    OV7670_CS = 0;  // 使能摄像头

    LCD_Clear(BLACK);

    // 主循环
    while (1) {
        // 通过串口传输图像数据到电脑端
        camera_refresh_1();  
        
        // 也可以在LCD上显示图像（可选）
        // camera_refresh();
    }
}
