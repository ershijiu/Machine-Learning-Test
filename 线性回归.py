import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#计算代价函数 J（θ）X为训练样本矩阵，其中第一列为1，相当于y=h(x)=θ_0+θ_1x(例如用面积预测房价样本里面的所有面积),y相当于训练样本里面的所有房价矩阵
#之所以把训练样本表示为矩阵，是为了利用矩阵运算的特性，不用循环一次性计算出训练样本在给定θ_0和θ_1值（θ_0和θ_1的值会在梯度下降算法循环中不断的更新）时代价函数的值
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

#梯度下降算法
#这个部分实现了Ѳ的更新
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost
    
if __name__ == '__main__':  
    path =  'ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    #展示DATA的前五行
    print(data.head())
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    plt.show()

    data.insert(0, 'Ones', 1)
    # 初始化X和y
    cols = data.shape[1]
    X = data.iloc[:,:-1]#X是data里的除最后列
    y = data.iloc[:,cols-1:cols]#y是data最后一列

    print(X.head())
    print(y.head())

    #代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0]))
    #初始化一些附加变量 - 学习速率α和要执行的迭代次数
    alpha = 0.01
    iters = 1500
    #现在让我们运行梯度下降算法来将我们的参数θ适合于训练集。
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    #预测35000和70000城市规模的小吃摊利润
    predict1 = [1,3.5]*g.T
    print("predict1:",predict1)
    predict2 = [1,7]*g.T
    print("predict2:",predict2)
    #原始数据以及拟合的直线
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

