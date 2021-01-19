根据策略梯度方法，参数更新方程式为：
$$
\theta_{new}=\theta_{old}+\alpha\nabla_{\theta}J\tag{1}
$$

在策略梯度方法中，合适的步长$\alpha$对于参数更新至关重要，当步长不合适时，更新的参数所对应的策略是一个更不好的策略，当利用这个更不好的策略进行采样学习时，再次更新的参数会更差，因此很容易导致越学越差，最后崩溃。
TRPO要解决的问题就是解决这个问题，找到新的策略使得新的回报函数的值单调增，或单调不减。

$\pi$是一个随机策略，$\rho_0(s_0)$是初始状态$s_0$的分布。$\eta\left(\pi\right)$代表折扣奖赏的期望，定义如下：
$$
\eta\left(\pi\right)=E_{s_0,a_0,\cdots ~}\left[\sum_{t=0}^{\infty}{\gamma^tA_{\pi}\left(s_t,a_t\right)}\right]\tag{2}
\\ subject\ to~~~\ s_0∼\rho_0(s_0),a_t∼\pi(a_t|s_t),s_{t+1}∼P(s_{t+1}|s_t, a_t)
$$

TRPO的目的是找到新的策略，使得回报函数单调不减。那么如果将新的策略所对应的回报函数可以用旧的策略所对应的回报函数与其他项之和(公式3)代替并保证新的策略所对应的其他项大于等于零，那么新的策略就能保证回报函数单调不减。
$$
\eta\left(\tilde{\pi}\right)=\eta\left(\pi\right)+E_{s_0,a_0,\cdots ~\tilde{\pi}}\left[\sum_{t=0}^{\infty}{\gamma^tA_{\pi}\left(s_t,a_t\right)}\right]\tag{3}\\
$$

$$
其中, A_{\pi}\left(s,a\right)=Q_{\pi}\left(s,a\right)-V_{\pi}\left(s\right)\tag{4}
$$



证明如下($\tilde{\pi}$为新策略，$\pi$为旧策略)：
$$
E_{\tau |\tilde{\pi}}\left[\sum_{t=0}^{\infty}{\gamma^tA_{\pi}\left(s_t,a_t\right)}\right] \\ =E_{\tau |\tilde{\pi}}\left[\sum_{t=0}^{\infty}{\gamma^t\left(r\left(s\right)+\gamma V^{\pi}\left(s_{t+1}\right)-V^{\pi}\left(s_t\right)\right)}\right] \\ =E_{\tau |\tilde{\pi}}\left[\sum_{t=0}^{\infty}{\gamma^t\left(r\left(s_t\right)\right)+\sum_{t=0}^{\infty}{\gamma^t\left(\gamma V^{\pi}\left(s_{t+1}\right)-V^{\pi}\left(s_t\right)\right)}}\right] \\ =E_{\tau |\tilde{\pi}}\left[\sum_{t=0}^{\infty}{\gamma^t\left(r\left(s_t\right)\right)}\right]+E_{s_0}\left[-V^{\pi}\left(s_0\right)\right] \\ =\eta\left(\tilde{\pi}\right)-\eta\left(\pi\right)
$$

我们定义:
$$
\rho_{\pi}\left(s\right)=P\left(s_0=s\right)+\gamma P\left(s_1=s\right)+\gamma^2P\left(s_2=s\right)+\cdots\tag{5}
$$

为了出现策略项，我们可以利用公式（5）将公式（3）改写为
$$
\eta\left(\tilde{\pi}\right)=\eta\left(\pi\right)+\sum_{t=0}^{\infty}{\sum_s{P\left(s_t=s|\tilde{\pi}\right)}}\sum_a{\tilde{\pi}\left(a|s\right)\gamma^tA_{\pi}\left(s,a\right)}=\eta\left(\pi\right)+\sum_s{\rho_{\tilde{\pi}}\left(s\right)\sum_a{\tilde{\pi}\left(a|s\right)A^{\pi}\left(s,a\right)}}\tag{6}
$$

由于$\rho_{\tilde{\pi}}$严重的依赖于新的策略$\tilde{\pi}$，使得公式（6）很难去优化。因此，我们忽略因策略改变而产生的状态分布的改变，即令$\rho_{\pi}\approx \rho_{\tilde{\pi}}$，近似后的公式为
$$
L_{\pi}\left(\tilde{\pi}\right)=\eta\left(\pi\right)+\sum_s{\rho_{\pi}\left(s\right)\sum_a{\tilde{\pi}\left(a|s\right)A^{\pi}\left(s,a\right)}}\tag{7}
$$

对比公式（6）与公式（7），我们发现$L_{\pi}\left(\tilde{\pi}\right)\textrm{，}\eta\left(\tilde{\pi}\right)$在策略 $\pi_{\theta_{old}} $处一阶近似，即：
$$
L_{\pi_{\theta_{old}}}\left(\pi_{\theta_{old}}\right)=\eta\left(\pi_{\theta_{old}}\right) \\ \nabla_{\theta}L_{\pi_{\theta_{old}}}\left(\pi_{\theta}\right)|_{\theta =\theta_{old}}=\nabla_{\theta}\eta\left(\pi_{\theta}\right)|_{\theta =\theta_{old}}\tag{8}
$$

TRPO在“自然策略梯度”的基础上提出了如下的算法，
$$
\eta\left(\tilde{\pi}\right)\geqslant L_{\pi}\left(\tilde{\pi}\right)-CD_{KL}^{\max}\left(\pi ,\tilde{\pi}\right) \\ subject\ to~~~\ C=\frac{2\varepsilon\gamma}{\left(1-\gamma\right)^2},\varepsilon=\max_{s,a}|A_{\pi}\left(s,a\right)|\tag{9}
$$

该不等式带给我们重要的启示，那就是给出了$\eta\left(\tilde{\pi}\right)$的下界，我们定义这个下界为$M_i\left(\pi\right)=L_{\pi_i}\left(\pi\right)-CD_{KL}^{\max}\left(\pi_i,\pi\right)$

利用这个下界我们可以证明策略的单调性：

$$
\eta\left(\pi_{i+1}\right)\geqslant M_i\left(\pi_{i+1}\right)\\
\eta\left(\pi_i\right)=M_i\left(\pi_i\right)\\
则\ \eta\left(\pi_{i+1}\right)-\eta\left(\pi_i\right)\geqslant M_i\left(\pi_{i+1}\right)-M\left(\pi_i\right)\tag{10}
$$

如果新的策略$\pi_{i+1}$能使得$M_i$最大，那么有不等式$M_i\left(\pi_{i+1}\right)-M\left(\pi_i\right)\geqslant 0 $，则$\eta\left(\pi_{i+1}\right)-\eta\left(\pi_i\right)\geqslant 0 $，那么我们的目标将转化为寻找使得$M_i$最大的新的策略。可形式化为
$$
maximize_{\theta}\left[L_{\theta_{old}}\left(\theta\right)-CD_{KL}^{\max}\left(\theta_{old},\theta\right)\right]\tag{11}
$$

然而在实际中，使用惩罚系数 C 会使得更新步伐非常小，因此，提出如下的形式
$$
maximize_{\theta}L_{\theta_{old}}\left(\theta\right)\\
subject\ to~~~D_{KL}^{\max}\left(\theta_{old},\theta\right)\leqslant \delta\tag{12}
$$

但是，这个问题强加了一个约束，即KL散度在状态空间的每个点都有界限， 尽管理论上可行，但由于存在大量约束，这个问题难以解决。因此我们可以使用考虑平均KL散度来近似
$$
maximize_{\theta}L_{\theta_{old}}\left(\theta\right)\\
subject\ to~~~\bar{D}_{KL}^{\rho_{\theta_{old}}}\left(\theta_{old},\theta\right)\le\delta
$$

也即：
$$
maximize_{\theta}\sum_s{\rho_{\theta_{old}}\left(s\right)\sum_a{\pi_\theta\left(a|s\right)A_{\theta_{old}}\left(s,a\right)}}\\
subject\ to~~~\bar{D}_{KL}^{\rho_{\theta_{old}}}\left(\theta_{old},\theta\right)\le\delta\tag{13}
$$


另外，我们同样也可以引入“重要性采样”，并作形式上的演化，最终的不等式化为
$$
maximize_{\theta}\;E_{s~\pi_{\theta_{old}},a~\pi_{\theta_{old}}}\left[\frac{\pi_{\theta}\left(a|s\right)}{\pi_{\theta_{old}}\left(a|s\right)}A_{\theta_{old}}\left(s,a\right)\right] \\ subject\ to\;
E_{s~\pi_{\theta_{old}}}\left[D_{KL}\left(\pi_{\theta_{old}}\left(\cdot |s\right)||\pi_{\theta}\left(\cdot |s\right)\right)\right]\le\delta\tag{14}
$$

论文中提出，可以将$A_{\theta_{old}}\left(s,a\right)$用$Q_{\theta_{old}}\left(s,a\right)$代替。

论文的部分技巧总结
* 理论上证明了可通过优化替代目标$M_i\left(\pi\right)$并对KL散度进行惩罚来更新策略使得$\eta$单调递增。 然而，较大惩罚系数$C$会导致更新步伐过小，所以我们希望减小这个系数。实际上，很难有力地选择惩罚系数，因此我们使用硬约束$\delta$（KL散度的界限）而不是惩罚。
* $D_{KL}^{\max}\left(\theta_{old},\theta\right)$很难进行数值优化和估计，因此我们用$\bar{D}_{KL}^{\rho_{\theta_{old}}}\left(\theta_{old},\theta\right)$来代替。
* 我们的理论忽略了优势函数的估计误差。 Kakade和Langford（2002）在他们的推导中考虑了这个误差，并且在本文的背景中也存在相同的论点，但是为了简单化我们省略了它。
