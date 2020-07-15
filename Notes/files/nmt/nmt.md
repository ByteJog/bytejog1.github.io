### 注意力模型直观理解（Attention Model Intuition）

#### 1. 注意力模型（Attention Model）

![attention_Model_Intuition](D:/BIJI/BIJI OF NLP/序列模型(Sequence Models)/image/attention_Model_Intuition.jpg)

$a^{<t'>}=\left( \begin{array}{c} \overrightarrow{a}^{<t'>} \\ \overleftarrow{a}^{<t'>} \end{array} \right)$ 为双向RNN激活值的组合

$\alpha^{<t,t'>}$ 代表了 $y^{<t>}$ 需要给 $a^{<t'>}$ 的注意力大小

在每一个时间步 t, $\sum_{t'}\alpha^{<t,t'>}=1$

$c^{<t>}=\sum_{t'}\alpha^{<t,t'>}a^{<t'>}$

$\alpha^{<t,t'>}$ 的获取在下一节讲述

#### 2. 注意力$\alpha^{<t,t'>}$的获取

![Attention Model1](D:/BIJI/BIJI OF NLP/序列模型(Sequence Models)/image/Attention Model1.png)

$\alpha^{<t,t'>}=\cfrac{exp(e^{<t,t'>})}{\sum_{t'=1}^{Tx} exp(e^{<t,t'>})}$

![Attention Model2](D:/BIJI/BIJI OF NLP/序列模型(Sequence Models)/image/Attention Model2.png)

#### 3. 带注意力的神经机器翻译示意图

下图是带注意力的神经机器翻译示意图，以提醒您模型的工作原理。 左侧的图表展示注意力模型，右侧的图表展示了一个“注意”步骤。

<table>
<td> 
<img src="image/attn_model.png" style="width:300;height:300px;"> <br>
</td> 
<td> 
<img src="image/attn_mechanism.png" style="width:300;height:300px;"> <br>
</td> 
</table>

