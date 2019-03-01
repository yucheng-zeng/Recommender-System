import random
import math

'''
def LearningLFM(train, F, n, alpha, lambda):
    [p,q] = InitLFM(train, F)
    for step in range(0, n):
        for u,i,rui in train.items():
            pui = Predict(u, i, p, q)
            eui = rui - pui
            for f in range(0,F):
                p[u][k] += alpha * (q[i][k] * eui - lambda * p[u][k])
                q[i][k] += alpha * (p[u][k] * eui - lambda * q[i][k])
        alpha *= 0.9
    return list(p, q)

def InitLFM(train, F):
    p = dict()
    q = dict()
    for u, i, rui in train.items():
        if u not in p:
            p[u] = [random.random()/math.sqrt(F) for x in range(0,F)]
        if i not in q:
            q[i] = [random.random()/math.sqrt(F) for x in range(0,F)]
    return list(p, q)

def Predict(u, i, p, q):
    return sum(p[u][f] * q[i][f] for f in range(0,len(p[u]))
'''







# 初始化
def InitBiasLFM(train, F):
    """
    :param train: 训练数据集
    :param F: 隐类格式,选择前多少个奇异值
    :return: 初始化后的bu,bi,p,q
    """
    p = dict()  # 用户-奇异值矩阵
    q = dict()  # 物品-奇异值矩阵
    bu = dict()  # 用户偏置项
    bi = dict()  # 物品偏置项
    for user_item, rui in train.items():
        user = user_item[0]
        item = user_item[1]
        bu[user] = 0
        bi[item] = 0
        if user not in p:
            p[user] = [random.random() / math.sqrt(F) for x in range(0, F)]
        if item not in q:
            q[item] = [random.random() / math.sqrt(F) for x in range(0, F)]
    return list(bu, bi, p, q)


# 偏置的隐语义模型
def LearningBiasLFM(train, F, n, alpha, lamb, mu):
    """
    :param train: 训练数据集
    :param F: 隐类格式,选择前多少个奇异值
    :param n: 迭代次数
    :param alpha: 学习率
    :param lamb: 正则化参数
    :param mu: 许训练集中所有评分的全局平均数
    :return: bu, bi, p, q
    """
    [bu, bi, p,q] = InitBiasLFM(train, F)  # 初始化
    for step in range(0, n):  # 逐步迭代
        for user_item,rui in train.items():  # 遍历训练数据集
            user = user_item[0]  # 获取当前记录的用户
            item = user_item[1]  # 获取当前记录的物品
            pui = Predict(user, item, p, q, bu, bi, mu)  # 当前模型预测 用户对物品的评分
            eui = rui - pui  # 评分损失
            bu[user] += alpha * (eui - lamb * bu[user])  # 更新用户偏置
            bi[item] += alpha * (eui - lamb * bi[item])  # 更新物品偏置
            for k in range(0, F):
                p[user][k] += alpha * (q[item][k] * eui - lamb * p[user][k])  # 更新用户矩阵
                q[item][k] += alpha * (p[user][k] * eui - lamb * q[item][k])  # 更新物品矩阵
        alpha *= 0.9  # 缩小学习率
    return list(bu, bi, p, q)  # 返回生成模型

# 预测用户评分
def Predict(user, item, p, q, bu, bi, mu):
    """
    :param user: 当前用户
    :param item: 当前物品
    :param p: 用户矩阵
    :param q: 物品矩阵
    :param bu: 用户偏置
    :param bi: 物品偏置
    :param mu: 全局平均数
    :return:
    """
    ret = mu + bu[user] + bi[item]
    ret += sum(p[user][f] * q[item][f] for f in range(0, len(p[user])))
    return ret



























def InitBiasLFM(train, F):
    p = dict()
    q = dict()
    bu = dict()
    bi = dict()
    for user_item, rui in train.items():
        user = user_item[0]
        item = user_item[1]
        bu[user] = 0
        bi[item] = 0
        if user not in p:
            p[user] = [random.random()/math.sqrt(F) for x in range(0,F)]
        if item not in q:
            q[item] = [random.random()/math.sqrt(F) for x in range(0,F)]
    return list(bu, bi, p, q)


def Predict(u, i, p, q, bu, bi, mu):
    ret = mu + bu[u] + bi[i]
    ret += sum(p[u][f] * q[i][f] for f in range(0, len(p[u]))
    return ret



# 加入领域的LFM
def LearningBiasLFM(train_ui, F, n, alpha, lambda, mu):
    [bu, bi, p, q, y] = InitLFM(train, F)
    z = dict()
    for step in range(0, n):
        for u,items in train_ui.items():
            z[u] = p[u]
            ru = 1 / math.sqrt(1.0 * len(items))
            for i,rui in items.items():
                for f in range(0,F):
                    z[u][f] += y[i][f] * ru
            sum = [0 for i in range(0,F)]
            for i,rui in items.items():
                pui = Predict()
                eui = rui - pui
                bu[u] += alpha * (eui - lambda * bu[u])
                bi[i] += alpha * (eui - lambda * bi[i])
                for f in range(0,F):
                    sum[k] += q[i][k] * eui * ru
                    p[u][k] += alpha * (q[i][k] * eui - lambda * p[u][k])
                    q[i][k] += alpha * ((z[u][k] + p[u][k]) * eui - lambda * q[i][k])
                    for i,rui in items.items():
            for f in range(0,F):
                y[i][f] += alpha * (sum[f] - lambda * y[i][f])
        alpha *= 0.9
    return list(bu, bi, p, q)
