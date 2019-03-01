import math
import random
import fileOperator as FO
import operator
import os
# 初始化
def InitBiasLFM(trainDataSet, testDataSet, F):
    """
    :param train: 训练数据集
    :param F: 隐类格式,选择前多少个奇异值
    :return: 初始化后的bu,bi,p,q
    """
    dataSet = {**trainDataSet, **testDataSet}
    p = dict()  # 用户-奇异值矩阵
    q = dict()  # 物品-奇异值矩阵
    bu = dict()  # 用户偏置项
    bi = dict()  # 物品偏置项
    for user_item, rui in dataSet.items():
        user = user_item[0]
        item = user_item[1]
        bu[user] = 0
        bi[item] = 0
        if user not in p:
            p[user] = [random.random() / math.sqrt(F) for x in range(0, F)]
        if item not in q:
            q[item] = [random.random() / math.sqrt(F) for x in range(0, F)]
    return bu, bi, p, q


# 偏置的隐语义模型
def LearningBiasLFM(trainDataSet, testDataSet, F, n, alpha, lamb, mu):
    """
    :param train: 训练数据集
    :param F: 隐类格式,选择前多少个奇异值
    :param n: 迭代次数
    :param alpha: 学习率
    :param lamb: 正则化参数
    :param mu: 许训练集中所有评分的全局平均数
    :return: bu, bi, p, q
    """
    bu, bi, p, q = InitBiasLFM(trainDataSet, testDataSet, F)  # 初始化
    for step in range(0, n):  # 逐步迭代
        for user_item,rui in trainDataSet.items():  # 遍历训练数据集
            user = int(user_item[0])  # 获取当前记录的用户
            item = int(user_item[1])  # 获取当前记录的物品
            pui = Predict(user, item, p, q, bu, bi, mu)  # 当前模型预测 用户对物品的评分
            eui = rui - pui  # 评分损失
            bu[user] += alpha * (eui - lamb * bu[user])  # 更新用户偏置
            bi[item] += alpha * (eui - lamb * bi[item])  # 更新物品偏置
            #print(bu[user])
            #print(bi[item])
            for k in range(0, F):
                p[user][k] += alpha * (q[item][k] * eui - lamb * p[user][k])  # 更新用户矩阵
                q[item][k] += alpha * (p[user][k] * eui - lamb * q[item][k])  # 更新物品矩阵
        alpha *= 0.9  # 缩小学习率
        #print('step=%s,alpha=%s'%(step,alpha))
    return bu, bi, p, q  # 返回生成模型


# 预测用户评分
def Predict(user, item, p, q, bu, bi, mu):
    """
    :param user: 当前用户
    :param item: 当前物品
    :param p: 用户矩阵bu, bi, p, q
    :param q: 物品矩阵
    :param bu: 用户偏置
    :param bi: 物品偏置
    :param mu: 全局平均数
    :return:
    """

    ret = mu + bu[user] + bi[item]
    if item in q.keys():
        for f in range(0, len(p[user])):
            ret += sum(p[user][f] * q[item][f] for f in range(0, len(p[user])))
    if ret < 1:
        ret = 1
    elif ret > 5:
        ret = 5
    return ret


def calMu(dataSet):
    valueMean = 0
    for value in dataSet.values():
        valueMean += value
    valueMean /= len(dataSet)
    return valueMean

def testSystem(F_list, step_list, LearnRating_list, penalty_list):
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')

    mu = calMu(trainDataSet)


    for F in F_list:
        for step in step_list:
            for LearnRating in LearnRating_list:
                for penalty in penalty_list:
                    bu, bi, p, q = LearningBiasLFM(trainDataSet, testDataSet, F, step, LearnRating, penalty, mu)
                    error = 0
                    for user_item, rating in testDataSet.items():
                        predictRating = Predict(user_item[0], user_item[1],p,q,bu,bi,mu)
                        error += (predictRating - rating)**2
                    RMSE = math.sqrt(error/len(testDataSet))
                    print('RMSE=%s' % RMSE)
                    with open('result.txt','a') as fileObject:
                        fileObject.write(str(F)+',')
                        fileObject.write(str(step)+',')
                        fileObject.write(str(LearnRating)+',')
                        fileObject.write(str(penalty)+',')
                        fileObject.write(str(mu)+',')
                        fileObject.write(str(RMSE)+'\n')


def initializeSystem(F_list, step_list, LearnRating_list, penalty_list):
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')
    dataSet = {**trainDataSet, **testDataSet}
    mu = calMu(dataSet)
    for F in F_list:
        for step in step_list:
            for LearnRating in LearnRating_list:
                for penalty in penalty_list:
                    print(F,step,LearnRating,penalty)
                    bu, bi, p, q = LearningBiasLFM(dataSet, dict(), F, step, LearnRating, penalty, mu)
                    path = 'SVD/Model_Parameter/' + str(F) + '-' + str(step) + '-' + str(LearnRating) + '-' + str(penalty)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    FO.WriteBuDict(path, bu)
                    FO.WriteBiDict(path, bi)
                    FO.WritePDict(path, p, F)
                    FO.WriteQDict(path, q, F)


def chooseBestParameter():
    BestParameter = dict()
    with open('result.txt') as fileObject:
        for line in fileObject.readlines():
            if len(line) <1:
                continue
            arr = line.split(',')
            if math.isnan(float(arr[-1])):
                continue
            if float(arr[-1]) > 0.95:
                continue
            BestParameter[float(arr[-1])] = [int(arr[0]),int(arr[1]),float(arr[2]),float(arr[3])]
    #return dict(sorted(BestParameter.items(), key=operator.itemgetter(0))[:30])
    return dict(sorted(BestParameter.items(), key=operator.itemgetter(0)))


def predictSystem(trainDataSet, testDataSet, itemList, user, p, q, bu, bi, mu, N=40):
    dataSet = {**trainDataSet, **testDataSet}
    userRated = set([user_item[1] for user_item, value in dataSet.items() if user_item[0] == user])
    rank = dict()
    UnRatedList = itemList - userRated
    for item in UnRatedList:
        rank[item] = Predict(user,item,p,q,bu,bi,mu)
    print(rank)
    return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])


def ReadParameter(F, step, LearnRating, penalty):
    path = 'SVD/Model_Parameter/'+str(F)+'-'+str(step)+'-'+str(LearnRating)+'-'+str(penalty)
    p = FO.ReadPDict(path)
    q = FO.ReadQDict(path)
    bu = FO.ReadBudict(path)
    bi = FO.ReadBiDict(path)
    return bu, bi, p, q


def fillMatrix(F_list, step_list, LearnRating_list, penalty_list, N = 30):
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')
    userList = FO.readUserList('data/users.txt')
    itemList = FO.readItemList('data/movies.txt')  # 获取物品列表
    dataSet = {**trainDataSet, **testDataSet}
    mu = calMu(dataSet)


    for F in F_list:
        for step in step_list:
            for LearnRating in LearnRating_list:
                for penalty in penalty_list:
                    print(F, step, LearnRating, penalty)
                    path = 'SVD/Matrix/' + str(F) + '-' + str(step) + '-' + str(LearnRating) + '-' + str(penalty)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    bu, bi, p, q = ReadParameter(F, step, LearnRating, penalty)
                    for user in userList:
                        print(user)
                        rank = dict()
                        userRated = set([user_item[1] for user_item, value in dataSet.items() if user_item[0] == user])
                        UnRatedList = itemList - userRated
                        for item in UnRatedList:
                            rating = Predict(user, item, p, q, bu, bi, mu)
                            if math.isnan(float(rating)):
                                continue
                            rank[item] = round(rating)
                        if len(rank) == 0:
                            continue
                        if N > len(UnRatedList):
                            N = len(UnRatedList)
                        chooseList = random.sample(UnRatedList, N)
                        with open(path+'/new_ratings.txt', 'a') as fileObject:
                            for choose in chooseList:
                                fileObject.write(str(user) + '::' + str(choose) + '::' + str(rank[choose]) + '\n')


if __name__=='__main__':
    '''
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    #testDataSet = FO.readDataSet('testDataSet.txt')
    mu = calMu(trainDataSet)
    parameter_lsit = [2, 5, 0.1, 0.5]
    bu, bi, p, q = LearningBiasLFM(trainDataSet, 2, 5, 0.1, 0.5, mu)
    print("bu=",bu)
    print("bi=",bi)
    print("p=",p)
    print("q=",q)
    '''
    '''
    testSystem()
    '''
    '''
    BestParameter = chooseBestParameter()
    print(BestParameter)
    for key, value in BestParameter.items():
        print(key,value)
    '''
    '''
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')
    mu = calMu(trainDataSet)
    F_list = [2, 5, 10]
    step_list = [5, 10, 20, 30]
    LearnRating_list = [0.01, 0.05, 0.1]
    penalty_list = [0.01, 0.05, 0.1]
    for F in F_list:
        for step in step_list:
            for LearnRating in LearnRating_list:
                for penalty in penalty_list:
                    initializeSystem(trainDataSet,testDataSet, F, step, LearnRating, penalty, mu)
    '''

    '''
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')
    mu = calMu(trainDataSet)
    itemList = FO.readItemList('data/movies.txt')
    parameter_lsit = [5, 10, 0.05, 0.1]
    path = 'SVD/Model_Parameter/'+str(parameter_lsit[0])+'-'+str(parameter_lsit[1])+'-'+str(parameter_lsit[2])+'-'+str(parameter_lsit[3])
    p = FO.ReadPDict(path)
    q = FO.ReadQDict(path)
    bu = FO.ReadBudict(path)
    bi = FO.ReadBiDict(path)
    rank = predictSystem(trainDataSet,testDataSet,itemList,231,p,q,bu,bi,mu,N=100)
    for item, value in rank.items():
        print(item, value)
    '''

    '''
        rank = chooseBestParameter()
        for key, value in rank.items():
            print(key, value)
    '''

    '''
    F_list = [2, 5, 10]
    step_list = [5, 10, 20, 30]
    LearnRating_list = [0.01, 0.05, 0.1]
    penalty_list = [0.01, 0.05, 0.1]
    initializeSystem(F_list, step_list, LearnRating_list, penalty_list)
    '''
    '''
    F_list = [2]
    step_list = [5]
    LearnRating_list = [0.01]
    penalty_list = [0.1]
    fillMatrix(F_list,step_list,LearnRating_list,penalty_list)
    '''

    '''
    F_list = [2, 5, 10]
    step_list = [5, 10, 20, 30]
    LearnRating_list = [0.01, 0.05, 0.1]
    penalty_list = [0.01, 0.05, 0.1]
    testSystem(F_list, step_list, LearnRating_list, penalty_list)
    '''


    BestParameter = chooseBestParameter()
    print(BestParameter)
    for key, values in BestParameter.items():
        print(key,values)

    with open('best_result_bias.txt','w') as fileObject:
        for key, values in BestParameter.items():
            for value in values:
                fileObject.write(str(value)+',')
            fileObject.write(str(key)+'\n')
