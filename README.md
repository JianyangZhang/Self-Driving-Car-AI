此无人车AI项目使用的Deep Q-learning算法，是DeepMind在2013年发明的深度强化学习算法，将Q-learning的思想与神经网络算法结合，也算是现代强化学习算法的源头了。研究者用这个算法在2015年让计算机学会了49种Atari游戏，并在大部分游戏中击败了人类。从适用性上来讲，我们不需要告诉AI具体的规则，只要让它不断摸索，它就能慢慢从中找到规律，完成许多之前被认为只有人类能完成的智力活动。

既然是Q-learning和Deep learnng的结合，就先结合无人车AI来讨论什么是Q-learning。

Q-learning是一种强化学习算法，无人车需要根据当前状态来采取动作，获得相应的奖励之后，再去改进这些动作，使得下次再到相同的状态时，无人车能做出更优的选择。我们用Q(S,A)表示在S状态时，采取A动作所获得的奖励(Rewards)，下面用字母R代表奖励，S'代表采取A动作后到达的新位置。
伪代码如下：
```
Initialize Q arbitrarily // 随机初始化Q值
Repeat (for each episode): // 每一次尝试，从车子出发到撞毁是一个episode
	Initialize S //车辆出发，S为初始位置的状态
	Repeat (for each step of episode):
		Q(S,A) ← (1-α)*Q(S,A) + α*[R + γ*maxQ(S',a)] // Q-learning核心贝尔曼方程，更新动作奖励
		S ← S' // 更新位置
	until S is terminal // 位置到达终点
```
贝尔曼方程(Bellman Equation)中，γ为折扣因子，α为学习速率。
γ越大，无人车会越重视以前的经验，越小就更重视眼前利益。α取值范围为0~1，取值越大，保留之前训练的效果就越少。

然后我们将Q-learning算法与深度学习结合。从High Level来看，Q-learning已经实现无人车基本的躲避路障功能，而深度学习算法可以让无人车自动总结并学习特征，减少人为设定特征的不完备性，以更好的适应非常复杂的环境条件。

首先，用一个深度神经网络来作为Q值的网络，地图上每个点有坐标(X1, X2)，将此状态输入神经网络来预测每个方向的Q值(图中假设有四个actions对应四个方向，所以一共得到4个新的Q值)。Q-target表示上一次到达该状态时所得到的Q值，然后使用均方差(mean-square error)来定义Loss Function。
![cnn](https://user-images.githubusercontent.com/22739177/32696393-b821cd68-c72b-11e7-8789-cdab520b49ba.PNG)
计算出的L值被反馈(back-propagation)以计算每个突触(绿色圈圈)的权重w，重复此过程使此函数收敛(L值接近0)，就可以让无人车做出此状态下的最优选择。
