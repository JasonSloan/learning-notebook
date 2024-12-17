原始视频: bilibili西湖大学赵世钰<<强化学习>>

# 一. 基本概念

## 1. grid-world example

![](codes/assets/1.jpg)

网格分为: 可达、禁止、目标单元格，边界

## 2. State和State space

![](codes/assets/2.jpg)

## 3. Action

![](codes/assets/3.jpg)

Action space是和state有关的， 不同的state会有不同的Action space， 写作 $A(S_i)$

## 4. State transition

![](codes/assets/4.jpg)

## 5. Forbidden area

![](codes/assets/5.jpg)

## 6. Tabular respresentation

![](codes/assets/6.jpg)

这种表示方式只能表示确定性（deterministic）的例子，一般不用

## 7. State transition probability

![](codes/assets/7.jpg)

使用条件概率来表示状态转移： 

当前在s1状态，采取a2动作，下一步在s2状态的概率为1；

当前在s1状态，采取a2动作，下一步在si(i不等于2)状态的概率为0；

## 8. Policy

![](codes/assets/8.jpg)

![](codes/assets/9.jpg)

**在强化学习中，我们使用 $\pi$ 来表示策略。在从一个状态转移到另一个状态的时候，采取不同的动作的概率之和应为1。**

![](codes/assets/10.jpg)

![](codes/assets/11.jpg)

## 9. Reward

![](codes/assets/12.jpg)

![](codes/assets/13.jpg)

![](codes/assets/14.jpg)

![](codes/assets/15.jpg)

reward只依赖于当前状态和采取的动作，不取决于它下一刻处于什么状态

## 10. Trajectory and return

![](codes/assets/16.jpg)

## 11. Discounted return

![](codes/assets/17.jpg)

![](codes/assets/18.jpg)

## 12. Episode

![](codes/assets/19.jpg)

![](codes/assets/20.jpg)

# 二. Markov decision process(MDP)

![](codes/assets/21.jpg)

![](codes/assets/22.jpg)

# 三. 贝尔曼公式

## 1. 引出

![](codes/assets/23.jpg)

![](codes/assets/24.jpg)

![](codes/assets/25.jpg)

在上面的公式中， r已知、$\gamma$ 已知、P已知，则v可求

## 2. 公式推导

![](codes/assets/26.jpg)

![](codes/assets/27.jpg)

![](codes/assets/28.jpg)

![](codes/assets/29.jpg)

![](codes/assets/30.jpg)

![](codes/assets/31.jpg)

**注意： 这里的$\pi(a|s)$指的是当前状态为是s，采取动作a的概率**

![](codes/assets/32.jpg)

![](codes/assets/33.jpg)

![](codes/assets/34.jpg)

**贝尔曼公式解释：** 在状态s时，对于一个给定的策略$\pi$，所能获得的state value值为两部分，第一部分是即时奖励，第二部分是未来奖励。

其中，$ \pi ( a | s )$含义为当前状态为s, 采取动作a的概率， $\sum _ { a }$含义为遍历所有可能得动作a；

$p(r|s, a)$含义为当前所在状态为s, 且采取的动作为a，获得的reward值为r的概率为$p(r|s, a)$，$\sum _ { r }$含义为遍历所有可能获得的reward值r；

$p(s'|s, a)$含义为当前所在状态为s采取动作a下一时刻跳转到状态s'的概率为$p(s'|s, a)$。$\sum _ { s' }$含义为遍历所有下一时刻可能到达的状态s', $v_\pi(s')$含义为下一时刻到达s'时的state value值。

![](codes/assets/35.jpg)

这里求解的步骤略过

## 3. 公式的向量形式

![](codes/assets/36.jpg)

![](codes/assets/37.jpg)

≜符号在数学上的含义为“等价于”

![](codes/assets/38.jpg)

$p_{\pi}(s_j|s_i)$的含义为从状态$s_i$跳到状态$s_j$的概率，看下面的例子更清晰

![](codes/assets/39.jpg)

![](codes/assets/40.jpg)

## （插入）Policy evaluation概念

![](codes/assets/41.jpg)

![](codes/assets/42.jpg)

注意这里的$v_k$是向量，先假设一个$v_0$向量值，然后可以一直递归的求解$v_2 v_3 v_4$......，当k趋近于无穷大的时候，求得的序列$\{v_k\}$向量就等价于原始的$v_\pi$（证明略）

**代码实现见demo-bellman.ipynb**

![](codes/assets/43.jpg)

![](codes/assets/45.jpg)

## 4. action value

![](codes/assets/46.jpg)

![](codes/assets/47.jpg)

**已知$q_\pi(s, a)$(action value)求解$v_\pi(s)$(state value)公式解释：**在某个状态s处的state value值$v_\pi(s)$等于在该状态对可能采取的动作a的概率乘以在该状态采取动作a后得到的action value之积的和，也就是所有action value的期望值。

![](codes/assets/48.jpg)

**已知$v_\pi(s)$(state value)求解$q_\pi(s, a)$(action value)公式解释：**

公式第一项：$p(r|s, a)$含义为当前所在状态为s, 且采取的动作为a，获得的reward值为r的概率为$p(r|s, a)$。因此第一项含义为当前在状态s采取动作a后能获得的即时reward的期望值。

公式第二项： $p(s'|s, a)$当前所在状态为s采取动作a下一时刻跳转到状态s'的概率为$p(s'|s, a)$。因此第二项含义为当前在状态s采取动作a后能到达的所有状态s'的state value的期望值。

由上图中的公式（3）和公式（4）我们可以看出，如果在状态s处的action只有一个，且概率为1，那么$v_\pi(s)$就等于$q_\pi(s, a)$。

![](codes/assets/49.jpg)

![](codes/assets/50.jpg)

![](codes/assets/51.jpg)

#  三. 贝尔曼最优公式

## 1. 引入

![](codes/assets/52.jpg)

![](codes/assets/53.jpg)

![](codes/assets/54.jpg)

![](codes/assets/55.jpg)

$argmax_a$在数学上表示： 待求参数是a,  且要求是后面的表达式的值最大，返回值为a

![](codes/assets/56.jpg)

## 2. 公式定义与推导

![](codes/assets/57.jpg)

![](codes/assets/58.jpg)

![](codes/assets/59.jpg)

$\max_{\pi}$在数学上表示： 待求参数是$\pi$,  且要求是后面的表达式的值最大，返回值为后面的表达式的值

![](codes/assets/60.jpg)

![](codes/assets/61.jpg)

![](codes/assets/62.jpg)

![](codes/assets/63.jpg)

## 3. 公式求解

![](codes/assets/64.jpg)

![](codes/assets/65.jpg)

![](codes/assets/66.jpg)

上图中的定理就是为什么在求解贝尔曼公式时可以迭代求解的原因

![](codes/assets/67.jpg)

![](codes/assets/68.jpg)

![](codes/assets/69.jpg)

![](codes/assets/70.jpg)

![](codes/assets/71.jpg)

![](codes/assets/72.jpg)

## 4. 分析贝尔曼最优公式一些性质

![](codes/assets/73.jpg)

![](codes/assets/74.jpg)

![](codes/assets/75.jpg)

![](codes/assets/76.jpg)

![](codes/assets/77.jpg)

![](codes/assets/78.jpg)

![](codes/assets/79.jpg)

![](codes/assets/80.jpg)

![](codes/assets/81.jpg)

# 四、值迭代算法与策略迭代算法

## 1.值迭代算法求解贝尔曼最优公式策略 

![](codes/assets/82.jpg)

![](codes/assets/83.jpg)

![](codes/assets/84.jpg)

![](codes/assets/85.jpg)

![](codes/assets/86.jpg)

![](codes/assets/87.jpg)

![](codes/assets/88.jpg)

![](codes/assets/89.jpg)

**值迭代算法流程：**

定义好进入到每个state的reward值---->初始化v_pi---->根据初始化好的v_0值计算每个state的action value值（公式1），记作q_table---->根据q_table中记录的action value值，选取每个state位置的最大action value值对应的action更新策略（公式2）---->更新v_pi，更新方式为将每个state位置的最大action value值赋值给v_pi（公式3）---->计算每个state的action value值......

公式1：$q_k(s,a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v_k(s')$

公式2：$\pi_{k+1}(a|s)=\left\{\begin{array}{ll}1 & a=a_k^*(s) \\0 & a \neq a_k^*(s)\end{array}\right.$

公式3： $v_{k+1}(s) = \max_a q_k(a, s)$

**代码实现见： "codes/value-iteration-bellman.ipynb"**

## 2.策略迭代算法求解贝尔曼最优公式策略

![](codes/assets/91.jpg)

![](codes/assets/92.jpg)

![](codes/assets/93.jpg)

![](codes/assets/94.jpg)

![](codes/assets/95.jpg)

![](codes/assets/96.jpg)

![](codes/assets/97.jpg)

![](codes/assets/98.jpg)

![](codes/assets/99.jpg)

![](codes/assets/100.jpg)

![](codes/assets/101.jpg)

![](codes/assets/102.jpg)

**代码实现见： "codes/policy-iteration-bellman.ipynb"**

## 2.截断策略迭代算法求解贝尔曼最优公式策略

![](codes/assets/103.jpg)

![](codes/assets/104.jpg)

![](codes/assets/105.jpg)

![](codes/assets/106.jpg)

![](codes/assets/107.jpg)

# 五. 蒙特卡洛方法

## 1. 引出

![](codes/assets/108.jpg)

![](codes/assets/109.jpg)

![](codes/assets/110.jpg)

![](codes/assets/111.jpg)

![](codes/assets/112.jpg)

## 2. MC Basic

**蒙特卡洛算法的核心思想：**

我们求解policy的关键在于求解每个state处的action value，也就是$q_{{\pi}k}$，而求解$q_{{\pi}k}$的一种方式是根据state value和action value的转换公式（比如policy iteration算法），此种方式被称为model-base方法；而另一种方式就是直接根据定义，也就是在一个state处的$q_{{\pi}k}$，等于从这个state处出发可能产生的所有episode的return的期望，此种方式被称为model-free方法

![](codes/assets/113.jpg)

![](codes/assets/114.jpg)

![](codes/assets/115.jpg)

![](codes/assets/116.jpg)

![](codes/assets/117.jpg)

## 3. MC Basic example

![](codes/assets/118.jpg)

![](codes/assets/119.jpg)

![](codes/assets/120.jpg)

![](codes/assets/121.jpg)

![](codes/assets/123.jpg)

![](codes/assets/122.jpg)

![](codes/assets/124.jpg)

![](codes/assets/125.jpg)

![](codes/assets/126.jpg)

![](codes/assets/127.jpg)

## 4. MC Exploring Starts

![](codes/assets/128.jpg)

![](codes/assets/129.jpg)

![](codes/assets/130.jpg)

![](codes/assets/131.jpg)

![](codes/assets/132.jpg)

**MC-Exploring-Starts缺点很明显，就是要确保所有的state和所有的action都要被探索到，因为没被探索到的action有可能就是最优的action，而全部被探索往往是无法确保的，要确保的话，就需要从每一个(state, action)对儿出发产生episode，这样就又退化成了MC-Basic**

**解决上述问题的方案就是使用MC Epsilon-Greedy**

## 5. MC Epsilon-Greedy 

![](codes/assets/133.jpg)

![](codes/assets/134.jpg)

**greedy在这里的含义为**，当对某个state来说，如果最大的action value对应的action为a\*，那么当产生episode的时候，选择a\*的概率是最大的，但是也有一定比较小的概率选择其他的action，这样就保证了不会遗漏每一个action

![](codes/assets/135.jpg)

![](codes/assets/136.jpg)

![](codes/assets/137.jpg)

![](codes/assets/138.jpg)

## 6. MC Epsilon-Greedy examples

![](codes/assets/139.jpg)

![](codes/assets/140.jpg)

![](codes/assets/141.jpg)

![](codes/assets/142.jpg)

![](codes/assets/143.jpg)

![](codes/assets/144.jpg)

所以如果采用MC Epsilon算法时， 一般会先设置一个大的epsilon值进行探索，然后逐渐减小epsilon值直到0，以能获取最优策略

# 六. 随机近似理论与随机梯度下降

## 1. 引出

![](codes/assets/145.jpg)

![](codes/assets/146.jpg)

![](codes/assets/147.jpg)

![](codes/assets/148.jpg)

![](codes/assets/149.jpg)

![](codes/assets/150.jpg)

## 2. Robbins-Monro算法

![](codes/assets/151.jpg)

![](codes/assets/152.jpg)

![](codes/assets/153.jpg)

![](codes/assets/154.jpg)

注意，该公式和上面引出里的公式是一样的，随机近似理论

![](codes/assets/155.jpg)

**代码实现见： "codes/Robbins-Monro.ipynb"**

其实这个公式和SGD是一模一样的，在随机梯度下降时，上面的w为待求解参数，$a_k$为学习率，$g(w_k)$为目标函数的导数

![](codes/assets/156.jpg)

## 3. SGD

![](codes/assets/157.jpg)

![](codes/assets/158.jpg)

![](codes/assets/159.jpg)

![](codes/assets/160.jpg)

上面这个例子实际上目的是求w*，也就是求X的期望，这种情况就等同于mean estimation算法，所以说mean estimation算法是SGD的一种特殊情况。

![](codes/assets/161.jpg)

由上述化简出的公式也可以看出，和mean estimation算法一模一样

**SGD的性质：在距离最优点越远的时候，SGD的前进方向和GD越接近；在距离最优点越近的时候，SGD的前进方向和GD越不接近。因此SGD只会在收敛的时候会产生随机性，而在未收敛时，和GD一样，会有一个正确的前进方向。（证明课件略）**

![](codes/assets/162.jpg)

**代码实现见： "codes/SGD-MBGD-BGD.ipynb"**

![](codes/assets/164.jpg)

## 4. MBGD、BGD

![](codes/assets/163.jpg)

![](codes/assets/165.jpg)

![](codes/assets/166.jpg)

![](codes/assets/167.jpg)

![](codes/assets/168.jpg)

![](codes/assets/169.jpg)

# 七. 时序差分算法 

## 1. 引出

![](codes/assets/170.jpg)

![](codes/assets/171.jpg)

![](codes/assets/172.jpg)

## 2. TD learning

![](codes/assets/173.jpg)

$s_0, r_1, s_1, r_2......$代表的是在策略$\pi$下产生的一条episode

这里的$v_{t+?}(s_t)$代表的含义均是对$s_t$处的state value的估计，只不过是在这个episode上时，不同时刻对$s_t$处的state value的估计，比如，在t=3时刻，所在位置为$s_3$，利用公式那么对$s_3$的state value会有一次更新，它的state value值从$v_{3}(s_3)$变为$v_4(s_3)$，其余所有state处的state value均保持不变。

公式(2)的含义为，当使用TD learning算法更新state value时，只会更新当前所在state处的state value，其余所有的state value不会被更新

$v_t(s_t)$代表的含义为： 在已经选中一条episode后，当前时刻为t，当前所在state为$s_t$，当前对状态$s_t$的state value的估计值为$v_t(s_t)$

$v_{t+1}(s_t)$代表的含义为： 在这个episode上当前时刻为t+1，当前所在state为$s_{t+1}$，对上一时刻状态$s_t$的state value的估计值为$v_{t+1}(s_t)$

具体可以看Q-learning代码实现更直观

![](codes/assets/174.jpg)

![](codes/assets/175.jpg)

![](codes/assets/176.jpg)

![](codes/assets/177.jpg)

## 3. Sarsa

![](codes/assets/178.jpg)

![](codes/assets/179.jpg)

![](codes/assets/180.jpg)

![](codes/assets/181.jpg)

![](codes/assets/182.jpg)

![](codes/assets/183.jpg)

![](codes/assets/185.jpg)

total reward为什么是趋近于0而不是大于0，因为这是使用了epsilon-greedy的方法，也就是对每个state，还是会有一定的概率选择非最优的action，所以还会引入负的reward

## 4. Expected Sarsa(不重要)

![](codes/assets/186.jpg)

![](codes/assets/187.jpg)

## 5. n-step Sarsa(不重要)

![](codes/assets/188.jpg)

![](codes/assets/189.jpg)

![](codes/assets/190.jpg)

## 6. Q-learning

![](codes/assets/191.jpg)

![](codes/assets/192.jpg)

![](codes/assets/193.jpg)

On-policy含义为：在生成一个episode的经验的过程中，改进当前策略

Off-policy含义为：先生成一个episode的经验，然后根据该经验改进当前策略

![](codes/assets/194.jpg)

![](codes/assets/195.jpg)

![](codes/assets/196.jpg)

![](codes/assets/197.jpg)

![](codes/assets/198.jpg)

![](codes/assets/199.jpg)

![](codes/assets/200.jpg)

注意off-policy版本的这里的$\pi_b$和$\pi_T$是两个不同的策略，$\pi_b$用来生成数据（产生episode），$\pi_T$是待优化的目标策略

![](codes/assets/202.jpg)

![](codes/assets/203.jpg)

![](codes/assets/204.jpg)

![](codes/assets/205.jpg)

## 7. TD-learning系列算法总结

![](codes/assets/206.jpg)

![](codes/assets/207.jpg)

# 八. 值函数近似

## 1. 引出

![](codes/assets/208.jpg)

![](codes/assets/209.jpg)

![](codes/assets/210.jpg)

这里的T代表转置的意思，一般，一个一维的向量默认为列向量

![](codes/assets/211.jpg)

![](codes/assets/212.jpg)

![](codes/assets/213.jpg)

## 2. 通过值函数近似估计state value

![](codes/assets/214.jpg)

![](codes/assets/215.jpg)

如果是数据是均匀分布，那么这里的目标函数实际上就等同于深度学习里的MSE

![](codes/assets/216.jpg)

而实际上在强化学习中，数据分布是不均匀的（比如网格世界中某些state会被频繁访问，而有些几乎不会被访问），所以目标函数实际上应该是一个带权重的MSE

![](codes/assets/217.jpg)

![](codes/assets/218.jpg)

![](codes/assets/219.jpg)

![](codes/assets/220.jpg)

![](codes/assets/221.jpg)

![](codes/assets/222.jpg)

![](codes/assets/223.jpg)

![](codes/assets/224.jpg)

![](codes/assets/225.jpg)

![](codes/assets/226.jpg)

![](codes/assets/227.jpg)

## 3. 例子

![](codes/assets/228.jpg)

![](codes/assets/229.jpg)

![](codes/assets/230.jpg)

![](codes/assets/231.jpg)

![](codes/assets/232.jpg)

![](codes/assets/233.jpg)

![](codes/assets/234.jpg)

![](codes/assets/235.jpg)

## 4. 值函数近似Sarsa

![](codes/assets/236.jpg)

![](codes/assets/237.jpg)

![](codes/assets/238.jpg)

## 5. 值函数近似Q-learning

![](codes/assets/239.jpg)

![](codes/assets/240.jpg)

![](codes/assets/241.jpg)

## 6. Deep Q-learning

![](codes/assets/242.jpg)

![](codes/assets/243.jpg)

![](codes/assets/244.jpg)

![](codes/assets/245.jpg)

基本思想就是先初始化两个网络，参数相同，一个用来计算上图中的红色的action value，另一个用来计算上图中蓝色的action value；然后在训练过程中，先固定住红色网络的参数，更新蓝色网络参数，一定轮次后将蓝色网络的参数赋值给红色网络。如此循环往复

![](codes/assets/246.jpg)

![](codes/assets/247.jpg)

![](codes/assets/248.jpg)

![](codes/assets/249.jpg)

![](codes/assets/250.jpg)

![](codes/assets/251.jpg)

## 7. Deep Q-learning例子

![](codes/assets/253.jpg)

![](codes/assets/254.jpg)

![](codes/assets/255.jpg)

![](codes/assets/256.jpg)

![](codes/assets/257.jpg)

从上图可以看出，在强化学习中，即使损失函数一直在下降，也不能代表所估计的策略是好的策略，因为在数据不充足的情况下，神经网络逼近的只是当前数据下的'最优'

# 九. 策略梯度方法

## 1. 基本思路

![](codes/assets/258.jpg)

![](codes/assets/259.jpg)

![](codes/assets/260.jpg)

![](codes/assets/261.jpg)

![](codes/assets/262.jpg)

![](codes/assets/263.jpg)

## 2. 目标函数

![](codes/assets/264.jpg)

![](codes/assets/265.jpg)

![](codes/assets/266.jpg)

![](codes/assets/267.jpg)

![](codes/assets/268.jpg)

![](codes/assets/269.jpg)

![](codes/assets/270.jpg)

![](codes/assets/271.jpg)

![](codes/assets/272.jpg)

![](codes/assets/273.jpg)

![](codes/assets/274.jpg)

![](codes/assets/275.jpg)

## 3. 目标函数的梯度

![](codes/assets/276.jpg)

![](codes/assets/277.jpg)

![](codes/assets/278.jpg)

![](codes/assets/279.jpg)

![](codes/assets/280.jpg)

![](codes/assets/281.jpg)

![](codes/assets/282.jpg)

![](codes/assets/283.jpg)

## 4. 梯度上升算法REINFORCE

![](codes/assets/284.jpg)

![](codes/assets/285.jpg)

![](codes/assets/286.jpg)

![](codes/assets/287.jpg)

![](codes/assets/288.jpg)

![](codes/assets/289.jpg)

![](codes/assets/290.jpg)

![](codes/assets/291.jpg)







































































































































## 

















































































