#include "sys.h"  


//THUMB指令不支持汇编内联
//采用如下方法实现执行汇编指令WFI  
//__asm void WFI_SET(void)
//{
//	WFI;		  
//}
////关闭所有中断(但是不包括fault和NMI中断)
//__asm void INTX_DISABLE(void)
//{
//	CPSID   I
//	BX      LR	  
//}
////开启所有中断
//__asm void INTX_ENABLE(void)
//{
//	CPSIE   I
//	BX      LR  
//}
////设置栈顶地址
////addr:栈顶地址
//__asm void MSR_MSP(u32 addr) 
//{
//	MSR MSP, r0 			//set Main Stack value
//	BX r14
//}
void NVIC_Configuration()
{
	NVIC_InitTypeDef NVIC_InitStructure;
	
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);

	NVIC_InitStructure.NVIC_IRQChannel=EXTI1_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=1;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority=1;
	NVIC_InitStructure.NVIC_IRQChannelCmd=ENABLE;
	NVIC_Init(&NVIC_InitStructure);
}















