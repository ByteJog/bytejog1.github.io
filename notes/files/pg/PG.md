策略梯度方法（PG）是强化学习（RL）中经常使用的算法。基于值函数的DQN算法通过近似估算状态-动作值函数$Q(s,a)$来推断最优策略，而策略梯度方法则是直接优化策略。

### **策略梯度方法推导**

策略梯度方法的目标是找到一组最优的神经网络参数$\theta^*$最大化总收益函数关于轨迹分布的期望

$$
\theta^*=\max_\theta\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_tr(\mathbf{s}_t,\mathbf{a}_t)\right]......(1)
$$

首先，定义我们的目标函数为：
$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_tr(\mathbf{s}_t,\mathbf{a}_t)\right]......(2)
$$

显然，直接求上式的梯度是不可能的，原因如下：
* 参数蕴含在每一时刻的 $\pi_{\theta}(a_t|s_t)$ 中；
* 策略会影响 $a_t$ 的概率分布，但不是直接的影响；
* $a_t$ 虽然具有概率分布，但是为了收获奖励，在实际环境中必须做出一个确定性的选择才行；
* 奖励本身与参数没有关系，因为奖励对参数求导为零。

因此，需要公式（2）变形，现在令轨迹的收益$r(\tau)=\sum_tr(\mathbf{s}_t,\mathbf{a}_t)$, 则目标函数可以写为
$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[r(\tau)]......(3)
$$
我们假设$\tau$的分布函数$p_\theta(\tau)$是可微分的，那么根据期望的定义，
$$
J(\theta)=\int p_\theta(\tau)r(\tau)\mathrm{d}\tau......(4)
$$
它的梯度为
$$
\nabla_\theta J(\theta)=\int \nabla_\theta p_\theta(\tau)r(\tau)\mathrm{d}\tau......(5)
$$

存在下面的一个恒等式：
$$
p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)=p_\theta(\tau)\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}=\nabla_\theta p_\theta(\tau)......(6)
$$

将该恒等式带入公式（5）得到
$$
\nabla_\theta J(\theta)=\int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)r(\tau)\mathrm{d}\tau=\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)r(\tau)]......(7)
$$

  <p style="color: red;">策略梯度可以表示为期望，这意味着我们可以使用抽样来近似它。</p>接下来我们谈谈如何计算$\nabla_\theta \log p_\theta(\tau)$

由Markov性，一条轨迹$\tau$出现的概率是
$$
p_\theta(\tau)=p(\mathbf{s}_1)\prod_{t=1}^T\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)p(\mathbf{s}_{t+1}|\mathbf{s}_t,\mathbf{a}_t)......(8)
$$

方程两边同时取对数，可得
$$
\log p_\theta(\tau)=\log p(\mathbf{s}_1)+\sum_{t=1}^T[\log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)+\log p(\mathbf{s}_{t+1}|\mathbf{s}_t,\mathbf{a}_t)]......(9)
$$

由于$\nabla_\theta \log p_\theta(\tau)$的值仅仅和带有参数$\theta$的项有关，那么
$$
\nabla_\theta\log p_\theta(\tau)=\sum_{t=1}^T\nabla_\theta\log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)......(10)
$$

最终，目标函数的梯度变为
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\left(\sum_{t=1}^T\nabla_\theta\log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)\right)\left(\sum_{t=1}^Tr(\mathbf{s}_t,\mathbf{a}_t)\right)\right]......(11)
$$

在从实际系统中抽样时，我们用下面的式子进行估算

$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\left[\left(\sum_{t=1}^T\nabla_\theta\log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})\right)\left(\sum_{t=1}^Tr(\mathbf{s}_{i,t},\mathbf{a}_{i,t})\right)\right]......(12)
$$


接下来，我们便可以使用 $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ 来更新参数 $\theta$

$\sum_{t=1}^T\nabla_\theta\log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)$是最大对数似然。在深度学习中，它衡量观察数据的可能性。在强化学习的背景下，它衡量了当前策略下轨迹的可能性。通过将其与奖励相乘，我们希望如果轨迹导致高的正奖励，则增加策略的出现可能性。相反，如果策略导致高的负奖励，我们希望降低该策略的出现可能性。

### **策略梯度方法改进**

**策略梯度方法的高方差问题：** 由于采样的轨迹千差万别，而且可能不同的 action 会带来一样的 Expected Reward。如果在分类任务中出现一个输入可以分为多个类的情况，梯度就会乱掉，因为网络不知道应该最大化哪个类别的输出概率。梯度很不稳定，就会带来收敛性很差的问题。为了解决这个问题，提出下面两种方法：
**1.修改因果关系：** 因果关系指的是，当前时间点的策略不能影响该时间点之前的时间点的所带来的收益，这个在直觉上很好理解，今天老板看到你工作努力想给你奖赏，老板不会给你昨天的工资加倍，只会给你今天的工资或者未来的工资加倍。

在公式（12）中便存在这样一个问题，在时间点 $t$ 的策略影响到了时间点 $t$ 之前的时间点 $t'$ 的收益。因此，对公式（12）做出如下调整
$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\left[\nabla_\theta\log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})\left(\sum_{t'=t}^Tr(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})\right)\right]......(13)
$$

其中，我们把$\hat{Q}_{i,t}=\sum_{t'=t}^Tr(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})$称作 *“reward-to-go”*，意为“之后的奖赏”。

**2.引入基线：** 首先，引入基线后的梯度的形式为
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)(r(\tau)-b)]......(14)
$$

其中，$b$ 是一个常数。接下来在数学上证明其合理性
* 计算期望：
$$
\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)b]=\int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)b\mathrm{d}\tau=\int \nabla_\theta p_\theta(\tau)b\mathrm{d}\tau=b\nabla_\theta\int p_\theta(\tau)\mathrm{d}\tau=b\nabla_\theta1=0......(15)
$$
因此，引入常数 $b$ 之后，$\nabla_\theta J(\theta)$ 的值不会改变。其中一个不错的基准线是
$$
b_t=\frac{1}{N}\sum_iQ^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t})
$$

* 计算方差
引入常数 $b$ 之后，尽管期望值没有发生变化，但是如何保证方差减小呢？
方差的表达式为 $\text{Var}[X]=\mathbf{E}[X^2]-\mathbf{E}[X]^2$。而$\nabla_\theta J(\theta)=\mathbf{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)(r(\tau)-b)]$，其方差为：
$$
 \text{Var}=\mathbb{E}_{\tau\sim p_\theta(\tau)}[(\nabla_\theta \log p_\theta(\tau)(r(\tau)-b))^2]-\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)(r(\tau)-b)]^2=\mathbb{E}_{\tau\sim p_\theta(\tau)}[(\nabla_\theta \log p_\theta(\tau)(r(\tau)-b))^2]-\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)r(\tau)]^2......(16)
$$
令$g(\tau):=\nabla_\theta \log p_\theta(\tau)$，并 使 $\text{Var}$ 对 $b$ 的一阶导数为0，则
$$
\frac{\mathrm{d}\text{Var}}{\mathrm{d}b}=\frac{\mathrm{d}}{\mathrm{d}b}\mathbb{E}[g(\tau)^2(r(\tau)-b)^2]=\frac{\mathrm{d}}{\mathrm{d}b}\left(\mathbb{E}[g(\tau)^2 r(\tau)^2]-2b\mathbb{E}[g(\tau)^2 r(\tau)]+b^2\mathbb{E}[g(\tau)^2]\right)=0......(17)
$$
化简后得到
$$
-2\mathbb{E}[g(\tau)^2 r(\tau)]+2b^*\mathbb{E}[g(\tau)^2]=0......(18)
$$
求解得到 $b$ 的最优值
$$
b^*=\frac{\mathbb{E}[g(\tau)^2 r(\tau)]}{\mathbb{E}[g(\tau)^2]}......(19)
$$

**策略梯度方法的样本效率问题：** 策略梯度法是一个在线 (on-policy) 算法，这是因为在计算策略梯度$\nabla_\theta J(\theta)=\mathbf{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)r(\tau)]$的时候所用的数据都是在新的策略下采样得到的，这就要求每次梯度更新之后就要根据新的策略全部重新采样，并把之前的在旧策略下采样到的样本全都丢弃，这种做法对数据的利用率非常低，使得收敛的速度也极低。那么如何有效利用旧的样本呢？这就需要引入**重要性采样** 的概念。

* **重要性采样（Importance Sampling）** 

  * 重要性采样的原理是
$$
\mathbb{E}_{x\sim p(x)}[f(x)]=\int p(x)f(x)\mathrm{d}x=\int q(x)\frac{p(x)}{q(x)}f(x)\mathrm{d}x=\mathbb{E}_{x\sim q(x)}\left[\frac{p(x)}{q(x)}f(x)\right]......(20)
$$

将重要性采样的原理应用到我们的目标函数，则满足以下等式
$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[r(\tau)]=\mathbb{E}_{\tau\sim \bar{p}(\tau)}\left[\frac{p_\theta(\tau)}{\bar{p}(\tau)}r(\tau)\right]......(21)
$$

由公式（8）可知
$$
\frac{p_\theta(\tau)}{\bar{p}(\tau)}=\frac{p(\mathbf{s}_1)\prod_{t=1}^T\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)p(\mathbf{s}_{t+1}|\mathbf{s}_t,\mathbf{a}_t)}{p(\mathbf{s}_1)\prod_{t=1}^T\bar{\pi}(\mathbf{a}_t|\mathbf{s}_t)p(\mathbf{s}_{t+1}|\mathbf{s}_t,\mathbf{a}_t)}=\frac{\prod_{t=1}^T\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)}{\prod_{t=1}^T\bar{\pi}(\mathbf{a}_t|\mathbf{s}_t)}
$$

现在，我们求目标函数 $J(\theta')=\mathbf{E}_{\tau\sim p_\theta(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}r(\tau)\right]$ 的梯度，
$$
\nabla_{\theta'}J(\theta')=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\frac{\nabla_{\theta'}p_{\theta'}(\tau)}{p_{\theta}(\tau)}r(\tau)\right]=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}\nabla_{\theta'}\log p_{\theta'}(\tau)r(\tau)\right]=\mathbf{E}_{\tau\sim p_\theta(\tau)}\left[\left(\prod_{t=1}^T\frac{\pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t)}{\pi_{\theta}(\mathbf{a}_t|\mathbf{s}_t)}\right)\left(\sum_{t=1}^T\nabla_{\theta'}\log \pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t)\right)\left(\sum_{t=1}^Tr(\mathbf{s}_t,\mathbf{a}_t)\right)\right]
$$

最后像前文所述，通过修正因果关系和引入基线来减小方差
$$
\nabla_{\theta'}J(\theta')=\mathbf{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T\nabla_{\theta'}\log \pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t)\left(\prod_{t'=1}^t\frac{\pi_{\theta'}(\mathbf{a}_{t'}|\mathbf{s}_{t'})}{\pi_{\theta}(\mathbf{a}_{t'}|\mathbf{s}_{t'})}\right)\left(\sum_{t'=t}^Tr(\mathbf{s}_{t'},\mathbf{a}_{t'})-b\right)\right]
$$

但是，这种形式也是存在问题的，上式中中间那块连乘部分的数值，是关于T指数增长的，如果每个数都略小于1，而时间轴非常长，这个乘积最终将非常接近于0，这样梯度效果就会很差了。

为了解决这个问题，我们可以重写目标函数的形式，如下
$$
J(\theta)=\sum_{t=1}^T\mathbf{E}_{(\mathbf{s}_t,\mathbf{a}_t)\sim p_\theta(\mathbf{s}_t,\mathbf{a}_t)}[r(\mathbf{s}_t,\mathbf{a}_t)]
$$

进一步展开可得
$$
J(\theta)=\sum_{t=1}^T\mathbf{E}_{\mathbf{s}_t\sim p_\theta(\mathbf{s}_t)}\left[\mathbf{E}_{\mathbf{a}_t\sim\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)}[r(\mathbf{s}_t,\mathbf{a}_t)]\right]
$$

这样我们便可以在两个层面上做重要性抽样了，最终形式为
$$
J(\theta')=\sum_{t=1}^T\mathbf{E}_{\mathbf{s}_t\sim p_\theta(\mathbf{s}_t)}\left[\frac{p_{\theta'}(\mathbf{s}_t)}{p_{\theta}(\mathbf{s}_t)}\mathbf{E}_{\mathbf{a}_t\sim\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)}\left[\frac{\pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t)}{\pi_{\theta}(\mathbf{a}_t|\mathbf{s}_t)}r(\mathbf{s}_t,\mathbf{a}_t)\right]\right]
$$

但是同时又带来了一个新的问题，那就是需要知道在新的给定策略下某个时刻在某个状态的概率$p_{\theta'}(\mathbf{s}_t)$，我们一般将$\frac{p_{\theta'}(\mathbf{s}_t)}{p_{\theta}(\mathbf{s}_t)}$这一项忽略，因为当两个策略足够接近时，这个比值近似为1。

### **引入“折扣因子（Discount Factor）”**
引入“折扣因子”的目的是让奖赏r有权重地相加，让最开始收获的奖励有最大的权重，越往后面权重越小，因为距离当前状态的越近，影响越大。前边的公式也将相应地做出调整。

$$
\theta^* = \mathop{argmax}_\theta \mathop{E}_{\tau\sim p_\theta(\tau)} [\sum_t \gamma^t r(s_t, a_t)]
$$

策略梯度地公式修改为：
$$
\nabla_\theta J(\theta) =\cfrac{1}{N}\sum_{i=1}^N\sum_{t=1}^T \left[ \nabla_\theta\log \pi_\theta(a_{i,t}|s_{i,t}) [(\sum_{t'=t}^T \gamma^{t-t'}r(s_{i,t'}, a_{i,t'}))-b)\right]
$$


**参考**
1. [CS 294-112 at UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/)
2. [UC Berkeley RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

