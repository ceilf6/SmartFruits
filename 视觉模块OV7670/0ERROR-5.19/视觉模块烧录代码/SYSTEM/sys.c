#include "sys.h"  


//THUMBָ�֧�ֻ������
//�������·���ʵ��ִ�л��ָ��WFI  
//__asm void WFI_SET(void)
//{
//	WFI;		  
//}
////�ر������ж�(���ǲ�����fault��NMI�ж�)
//__asm void INTX_DISABLE(void)
//{
//	CPSID   I
//	BX      LR	  
//}
////���������ж�
//__asm void INTX_ENABLE(void)
//{
//	CPSIE   I
//	BX      LR  
//}
////����ջ����ַ
////addr:ջ����ַ
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















