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







































































## 

















































































