#! https://zhuanlan.zhihu.com/p/505538646
# 推荐系统学习笔记
<!-- vscode-markdown-toc -->
* 1. [协同过滤(Collaborative Filtering)](#CollaborativeFiltering)
	* 1.1. [基于邻域的协同过滤](#)
		* 1.1.1. [基于用户的User-CF](#User-CF)
		* 1.1.2. [基于商品的Item-CF](#Item-CF)
		* 1.1.3. [基于领域的评分预测](#-1)
		* 1.1.4. [基于二部图的协同过滤](#-1)
	* 1.2. [基于模型的协同过滤](#-1)
		* 1.2.1. [关联规则](#-1)
		* 1.2.2. [矩阵分解](#-1)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->
##  1. <a name='CollaborativeFiltering'></a>协同过滤(Collaborative Filtering)
###  1.1. <a name=''></a>基于邻域的协同过滤
####  1.1.1. <a name='User-CF'></a>基于用户的User-CF
- 用户$u$和用户$v$的兴趣相似度
  - $Jaccard公式$
  $$w_{uv}=\frac{\vert N(u)\cap N(v)\vert}{\vert N(u)\cup N(v)\vert}$$
  - $余弦公式$
  $$w_{uv}=\frac{\vert N(u)\cap N(v)\vert}{\sqrt{\vert N(u)\vert\vert N(v)\vert}}$$
  - 其中$N(u)$为用户$u$有正反馈的商品集合
- 用户$u$对领域用户购买商品$i$的兴趣度预测$p(u, i)$
  $$p(u,i) = \sum_{v\in S(u,K)}{w_{uv}*r_{vi}}$$
  - $S(u,K)$：与用户$u$最相似的$K$个用户
  - $w_{uv}$：用户$u$和用户$v$的兴趣相似度
  - $r_{ui}$：用户$u$对商品$i$的兴趣观测度（正反馈为$1$，负反馈为$0$）
#####  1.1.1. <a name=''></a>推荐过程
- 离线预处理
  - 计算用户之间的相似度矩阵
  - 确定每个用户的邻域
- 在线推荐：针对活跃用户$u$，确定推荐列表
  - 确定候选商品集：
  $$C(u) = \{i\vert i\notin N(u)\&i\in N(v)\&v\in S(u,K)\}$$
  - 预测兴趣度：
  $$p(u,i) = \sum_{v\in S(u,K)}{w_{uv}*r_{vi}}$$
#####  1.1.2. <a name='InverseUserFrequency'></a>逆用户频率（Inverse User Frequency）
- 基本思想：增加惩罚系数降低热门商品的预测值；不同用户对冷门商品具有相同倾向更能说明兴趣程度
- 惩罚系数：$f_{i} = ln(\frac{n}{n_i})$
- $n$为总用户数，$n_i$对商品$i$有正反馈的用户数
- 经过修正，用户的兴趣相似度为
$$w_{uv}=\frac{\sum_{i\in N(u)\cap N(v)}{ln(\frac{n}{n_i})}}{\vert N(u)\cup N(v)\vert}$$

####  1.1.2. <a name='Item-CF'></a>基于商品的Item-CF
- 商品$i$和商品$j$的相似度
  - $Jaccard公式$
  $$w_{ij}=\frac{\vert N(i)\cap N(j)\vert}{\vert N(i)\cup N(j)\vert}$$
  - $余弦公式$
  $$w_{ij}=\frac{\vert N(i)\cap N(j)\vert}{\sqrt{\vert N(i)\vert\vert N(j)\vert}}$$
  - 条件相似度
  $$w_{ij} = P(j\vert i) = \frac{\vert N(i)\cap N(j)\vert}{\vert N(i)\vert}$$
  - 其中$N(i)$为购买商品$i$的用户集
- 用户$u$对领域用户购买商品$i$的兴趣度预测$p(u, i)$
  $$p(u,i) = \sum_{j\in N(u)}{I(i\in S(j, K)) *w_{ij}*r_{uj}}$$
  - $S(j,K)$：与商品$j$最相似的$K$个商品
  - $w_{ij}$：商品$i$和商品$j$的相似度
  - $r_{uj}$：用户$u$对商品$j$的兴趣观测度（正反馈为$1$，负反馈为$0$）
#####  1.2.1. <a name='-1'></a>惩罚活跃用户
- 商品相似度修正
$$w_{ij}=\frac{\sum_{u\in N(i)\cap N(j)}{ln(\frac{m}{m_u})}}{\vert N(i)\cup N(j)\vert}$$
- 其中$m$表示商品总数，$m_u$表示用户$u$有正反馈的商品数

####  1.1.3. <a name='-1'></a>基于领域的评分预测
- 用户$u$与$v$的余弦相似度
$$w_{uv}= \frac{\sum_{i\in N(u)\cap N(v)}{r_{ui}r_{vi}}}{\sqrt{\sum_{i\in N(u)}{r_{ui}^{2}}\sum_{j\in N(v)}{r_{vj}^{2}}}}$$
  - 其中$N(u)$为用户$u$有评价过的商品集合，$r_{ui}$为用户$u$对商品$i$的评分
- 用户$u$与$v$的$pearson$相似度
$$w_{uv}= \frac{\sum_{i\in N(u)\cap N(v)}{(r_{ui}-\bar{r}_{u})(r_{vi}-\bar{r}_{v})}}{\sqrt{\sum_{i\in N(u)}{(r_{ui}-\bar{r}_{u})^{2}}\sum_{j\in N(v)}{(r_{vj}-\bar{r}_{v})^{2}}}}$$
  - $\bar{r}_{u}$为用户$u$的平均评分
- 基于用户的CF，评分预测
$$\hat{r}_{ui}=\frac{\sum_{v\in N_{i}(u)}w_{uv}r_{vi}}{\sum_{v\in N_{i}(u)}\vert w_{uv}\vert}$$
  - 修正后的评分预测
$$\hat{r}_{ui}=\bar{r}_{u} + \frac{\sum_{v\in N_{i}(u)}w_{uv}(r_{vi}-\bar{r}_{v})}{\sum_{v\in N_{i}(u)}\vert w_{uv}\vert}$$
- 基于商品的CF，评分预测
$$\hat{r}_{ui}=\frac{\sum_{j\in N_{u}(i)}w_{ij}r_{uj}}{\sum_{j\in N_{u}(i)}\vert w_{ij}\vert}$$
  - 修正后的评分预测
$$\hat{r}_{ui}=\bar{r}_{i} + \frac{\sum_{j\in N_{u}(i)}w_{ij}(r_{uj}-\bar{r}_{j})}{\sum_{j\in N_{u}(i)}\vert w_{ij}\vert}$$

####  1.1.4. <a name='-1'></a>基于二部图的协同过滤
#####  1.4.1. <a name='-1'></a>激活扩散
- 思路：根据用户偏好的传递性挖掘用户的潜在偏好
- 和标准协同过滤的区别：标准协同过滤相当于仅有一次传递的二部图协同过滤，考虑和用户$u$有类似偏好的用户$v$选择的商品
- 用户行为矩阵$R = (r_{ui})$
$$r_{ui}=\begin{cases} 1, & 用户u对商品i有正反馈\\ 0, & 其他 \end{cases}$$
- $k$次扩散后的矩阵
$$R^{2k+1} = R * (R^{T} * R)^{k}$$
- 例：

| 用户 | 商品 |
| :----: | :----: |
| A | b, d |
| B | a, b, c |
| C | a, b, d | 
| D | a, e |

| 用户\商品 | a | b | c | d | e |
|:----:|----|----|----|----|----|
| A | 0 | 1 | 0 | 1 | 0 |
| B | 1 | 1 | 1 | 0 | 0 |
| C | 1 | 1 | 0 | 1 | 0 |
| D | 1 | 0 | 0 | 0 | 1 |

$$
R = \left(
\begin{matrix}
0 & 1 & 0 & 1 & 0\\
1 & 1 & 1 & 0 & 0\\
1 & 1 & 0 & 1 & 0\\
1 & 0 & 0 & 0 & 1
\end{matrix}
\right)
$$
经过一次扩散后的结果

$$ R^{3} = R * (R^{T} * R) = 
\left(
\begin{matrix}
3 & 5 & 1 & 4 & 0\\
6 & 6 & 3 & 3 & 1\\
6 & 7 & 2 & 5 & 1\\
4 & 2 & 1 & 1 & 2
\end{matrix}
\right)
$$
经过两次扩散后的结果
$$ R^{3} = R * (R^{T} * R)^{2} = 
\left(
\begin{matrix}
24 & 30 & 9 & 21 & 3\\
37 & 39 & 15 & 24 & 7\\
40 & 45 & 15 & 30 & 7\\
20 & 17 & 7 & 10 & 6
\end{matrix}
\right)
$$
则推荐列表如下
|步数k\用户| A | B | C | D |
|:--:|:--:|:--:|:--:|:--:|
|0| b, d| a, b, c| a, b, d| a, e|
| 1 | a, c | d, e | c, e| b, c, d|
| 2 | a, c, e| d, e| c, e| b, d, c|

#####  1.4.2. <a name='-1'></a>物质扩散
- 物质扩散和激活扩散的区别：物质扩散保证了物质守恒
- **初始化：**对有正反馈的结点分配一个单位物质
- **物质扩散：**根据相邻关系，平均分配传递物质
- **生成推荐列表：**根据物质大小对候选商品进行排序
- 用户$u$获得的物质$b_{u}$
$$ b_{u}=\sum_{j=1}^{n}{r_{uj}\frac{s_{j}}{k(I_{j})}}$$
  - $S = (s_1, s_2, ..., s_n)$为各商品结点的物质分配
  - $k(I_j)$为各商品结点的出度，即与用户相连接的边数
- 用户行为矩阵$R = (r_{ui})$
$$r_{ui}=\begin{cases} 1, & 用户u对商品i有正反馈\\ 0, & 其他 \end{cases}$$
- $k$次扩散后的矩阵
$$S^{k} = S^{0} * (W)^{k}$$
- 初始化 $S^{0} = R$
- 状态转移矩阵$W = (w_{lj})$
$$w_{lj} = \frac{1}{k(I_j)}\sum_{u=1}^{m}\frac{r_{ul}r_{uj}}{k(U_u)}$$
  - 其中用户获得的物质$b_u$$$ b_{u}=\sum_{j=1}^{n}{r_{uj}\frac{s_{j}}{k(I_{j})}}$$
  - 商品结点从用户获得的资源
$$s'_{uj}=\sum_{u=1}^{m}{r_{uj}\frac{b_u}{k(U_u)}}=\sum_{j=1}^{n}{s_{ul}w_{lj}}$$

###  1.2. <a name='-1'></a>基于模型的协同过滤
**基于模型的协同过滤与基于邻域的协同过滤的区别：** 基于邻域的协同过滤中，有相似行为的用户对彼此的影响是相同的，仅靠做出相同行为的数量进行判断和排序；而基于模型的协同过滤中，量化考虑了用户$u$的行为$i$对用户$v$的行为$j$的影响大小
####  1.2.1. <a name='-1'></a>关联规则
##### 关联规则度量

|名称|描述|符号|
|:--:|--|--|
|置信度|选择A的条件下倾向于B的条件概率|$P(B\vert A)$|
|支持度|A和B同时选择的概率|$P(A\cap B)$|
|期望可信度|选择B的概率|$P(B)$|
|改善度|条件概率/概率|$P(B\vert A)/P(B)$|
##### 关联规则挖掘
- 找出频繁项集
  - 频繁项集：支持度大于最小支持度的所有集合
- 找出强关联规则：
  - 由频繁项集生成关联规则
  - 保留置信度大于最小置信度的关联规则
  - 置信度高可能表示负相关性强
- 其中**最小支持度**和**最小置信度**都是超参数，需要人为给定
##### 先验原理
- 若$A$是频繁项集，则它的所有子集都是频繁项集
- 若$A$是非频繁项集，则它的所有超集都是非频繁项集
- 示例：mooc推荐系统

![mooc推荐系统](https://raw.githubusercontent.com//ZengKunwei-sh/images/main/mooc/recommendation_system/beida.png)

- 改善度(提升度)
$$lift(A->B) = \frac{confidence(A->B)}{P(B)}=\frac{P(A\cap B)}{P(A)P(B)}$$
  - 相关度(改善度)
$$corr(A,B) = lift(A->B) = lift(B->A) = \frac{P(A\cap B)}{P(A)P(B)}$$
  - $corr(A,B)<1$ 负相关
  - $corr(A,B)=1$ 相互独立
  - $corr(A,B)>1$ 正相关
##### 基于关联规则的推荐
- 寻找强关联规则（离线）
  - 根据**支持度**寻找频繁项集，生成关联规则
  - 根据**置信度**和**改善度**过滤频繁项集，保留强关联规则
- 确定候选项目集（在线）
  - 根据用户的行为确定项集
  - 根据强关联规则，取得候选项集
- 排序推荐项集
  - 根据**置信度**对候选项集进行排序，生成推荐列表

####  1.2.2. <a name='-1'></a>矩阵分解
##### 奇异值分解（SVD）
- 任意矩阵$R_{m×n}$都可以分解为三个矩阵的乘积
$$R_{m×n}=U_{m×m}*\Sigma_{m×n}*V_{n×n}$$
  - $U_{m×m}$和$V_{n×n}$为正交矩阵，$\Sigma_{m×n}$为对角矩阵
  > [奇异值分解(SVD)](https://zhuanlan.zhihu.com/p/448767610)
  > [特征值分解与奇异值分解](https://zhuanlan.zhihu.com/p/69540876)
- 目的：
  - $SVD$分解后，取最大的$k$个特征值，实现对矩阵的**降维**
  - 利用前$k$个奇异值和特征向量近似描述原始评分矩阵
- 缺陷：
  - $SVD$需要完整的矩阵
  - 而用户评分矩阵是系数的，缺失大量信息
##### 隐语义模型LFM
- 任意矩阵$R_{m×n}$都可以分解为两个矩阵的乘积
$$R_{m×n}=X_{m×k}Y_{n×k}^{T}$$
  - $k=rank(R)$
  - 解释含义：将用户行为矩阵$R_{m×n}$分解为用户矩阵$X_{m×k}$和商品矩阵$Y_{n×k}^{T}$，其中$k$表示用$k$个线性无关向量表示出用户的特征
- 目标函数：
$$min_{P,Q}(\sum_{(u, v)\in S}{(r_{ui}-<p_{u}, q_{v}>)^{2}})$$
  - $r_{ui}$为用户$u$对商品$i$的评分
  - $\hat{r_{ui}}=<p_{u}, q_{v}>$为对评分的预测
- 加正则化项的目标函数
$$min_{P,Q}(\sum_{(u, v)\in S}{(r_{ui}-<p_{u}, q_{v}>)^{2}}+\lambda [\vert\vert P\vert\vert^{2} + \vert\vert Q\vert\vert^{2}])$$
- 随机梯度下降法求解目标函数
  - 随机初始化参数$P$和$Q$
  - 每次随机抽取一个样本$(u, i, r_{ui})$
  - 计算目标函数，并迭代参数
$$L_{ui}=\sum_{(u, v)\in S}{(r_{ui}-<p_{u}, q_{v}>)^{2}}+\lambda [\vert\vert P\vert\vert^{2} + \vert\vert Q\vert\vert^{2}]$$
    - $p'_{u} = p_{u} - \alpha(\frac{\partial L_{ui}}{\partial p_{u}})$
    - $q'_{i} = q_{i} - \alpha(\frac{\partial L_{ui}}{\partial q_{i}})$
  - 直到参数收敛，或达到最大迭代次数
- 交替最小二乘法求解目标函数
  - 误差方程
$$Error(w\vert X, y)=(Xw-y)^{T}(Xw-y)$$
  - 最优解$w=(X^{T}X)^{-1}X^{T}y$
  >推导过程：[最小二乘法](https://zhuanlan.zhihu.com/p/128083562)
  - 代入目标函数的参数，$y=r_{ui}$
    - 固定$P$，$q_{i}=(\sum_{u\vert (u, i)\in S}{p_{u}p_{u}^{T}}+\lambda E)^{-1}\sum_{u\vert (u, i)\in S}{p_{u}r_{ui}}$
    - 固定$Q$，$p_{u}=(\sum_{i\vert (u, i)\in S}{q_{i}q_{i}^{T}}+\lambda E)^{-1}\sum_{i\vert (u, i)\in S}{q_{i}r_{ui}}$
  - 直到参数收敛，或达到最大迭代次数