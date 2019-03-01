import random
import math
import fileOperator as FO
import operator
import os


def calMu(dataSet):
    valueMean = 0
    for value in dataSet.values():
        valueMean += value
    valueMean /= len(dataSet)
    return valueMean

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
    y = dict()
    for user_item, rui in dataSet.items():
        user = user_item[0]
        item = user_item[1]
        bu[user] = 0
        bi[item] = 0
        if user not in p:
            p[user] = [random.random() / math.sqrt(F) for x in range(0, F)]
        if item not in q:
            y[item] = [random.random() / math.sqrt(F) for x in range(0, F)]
            q[item] = [random.random() / math.sqrt(F) for x in range(0, F)]
    return bu, bi, p, q, y


# 加入领域的LFM
def LearningBiasLFM(user_items, trainDataSet, testDataSet, F, n, alpha, lamb, mu):
    bu, bi, p, q, y = InitBiasLFM(trainDataSet, testDataSet, F)
    z = dict()
    for step in range(0, n):
        for user, items in user_items.items():
            z[user] = p[user]
            # print(z[user])
            ru = 1 / math.sqrt(1.0 * len(items))
            # print(ru)
            for item_rui in items:
                item = item_rui[0]
                for f in range(0, F):
                    z[user][f] = z[user][f] + y[item][f] * ru
            sum = [0 for i in range(0,F)]
            for item_rui in items:
                item = item_rui[0]
                rui = item_rui[1]
                pui = Predict(user, item, p, q, bu, bi, mu)
                eui = rui - pui
                bu[user] += alpha * (eui - lamb * bu[user])
                bi[item] += alpha * (eui - lamb * bi[item])
                for f in range(0, F):
                    sum[f] += q[item][f] * eui * ru
                    p[user][f] += alpha * (q[item][f] * eui - lamb * p[user][f])
                    q[item][f] += alpha * ((z[user][f] + p[user][f]) * eui - lamb * q[item][f])
            for item, rui in items:
                for f in range(0, F):
                    y[item][f] += alpha * (sum[f] - lamb * y[item][f])
        alpha *= 0.9
    return bu, bi, p, q


def Predict(user, item, p, q, bu, bi, mu):
    ret = mu + bu[user] + bi[item]
    if item in q.keys():
        if user in p.keys():
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


def testSystem(F_list, step_list,LearnRating_list,penalty_list):
    # 建立 用户 到 物品 的倒排表
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')

    user_items = dict()
    for user_item, rating in trainDataSet.items():
        user = user_item[0]
        item = user_item[1]
        if user not in user_items:
            user_items[user] = list()
        user_items[user].append((item, rating))

    mu = calMu(trainDataSet)

    for F in F_list:
        for step in step_list:
            for LearnRating in LearnRating_list:
                for penalty in penalty_list:
                    bu, bi, p, q = LearningBiasLFM(user_items, trainDataSet, testDataSet, F, step, LearnRating, penalty, mu)
                    error = 0
                    for user_item, rating in testDataSet.items():
                        predictRating = Predict(user_item[0], user_item[1], p, q, bu, bi, mu)
                        error += (predictRating - rating) ** 2
                    RMSE = math.sqrt(error / len(testDataSet))
                    print('RMSE=%s' % RMSE)
                    with open('result_ItemCF.txt','a') as fileObject:
                        fileObject.write(str(F)+',')
                        fileObject.write(str(step)+',')
                        fileObject.write(str(LearnRating)+',')
                        fileObject.write(str(penalty)+',')
                        fileObject.write(str(mu)+',')
                        fileObject.write(str(RMSE)+'\n')


def predictSystem(trainDataSet, testDataSet, itemList, user, p, q, bu, bi, mu, N=30):
    dataSet = {**trainDataSet, **testDataSet}
    userRated = set([user_item[1] for user_item, value in dataSet.items() if user_item[0] == user])
    rank = dict()
    UnRatedList = itemList - userRated
    for item in UnRatedList:
        rank[item] = Predict(user, item, p, q, bu, bi, mu)
    print(rank)
    return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])


def chooseBestParameter():
    BestParameter = dict()
    with open('result_ItemCF.txt') as fileObject:
        for line in fileObject.readlines():
            if len(line) < 1 :
                continue
            arr = line.split(',')
            if math.isnan(float(arr[-1])):
                continue
            if float(arr[-1]) > 0.95:
                continue
            BestParameter[float(arr[-1])] = [int(arr[0]),int(arr[1]),float(arr[2]),float(arr[3])]
    #return dict(sorted(BestParameter.items(), key=operator.itemgetter(0))[:30])
    return dict(sorted(BestParameter.items(), key=operator.itemgetter(0)))


def initializeSystem(F_list, step_list, LearnRating_list, penalty_list):
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')
    dataSet = {**trainDataSet, **testDataSet}

    mu = calMu(dataSet)  #
    user_items = dict()
    for user_item, rating in dataSet.items():
        user = user_item[0]
        item = user_item[1]
        if user not in user_items:
            user_items[user] = list()
        user_items[user].append((item, rating))

    for F in F_list:
        for step in step_list:
            for LearnRating in LearnRating_list:
                for penalty in penalty_list:
                    print(F, step, LearnRating, penalty)
                    bu, bi, p, q = LearningBiasLFM(user_items, dataSet, dict(), F, step, LearnRating, penalty,mu)
                    path = 'SVD++/Model_Parameter/' + str(F) + '-' + str(step) + '-' + str(LearnRating) + '-' + str(penalty)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    FO.WriteBuDict(path, bu)
                    FO.WriteBiDict(path, bi)
                    FO.WritePDict(path, p, F)
                    FO.WriteQDict(path, q, F)


def ReadParameter(F, step, LearnRating, penalty):
    path = 'SVD++/Model_Parameter/'+str(F)+'-'+str(step)+'-'+str(LearnRating)+'-'+str(penalty)
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
                    print(F,step,LearnRating,penalty)
                    path = 'SVD++/Matrix/' + str(F) + '-' + str(step) + '-' + str(LearnRating) + '-' + str(penalty)
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
                        with open(path + '/new_ratings.txt', 'a') as fileObject:
                            for choose in chooseList:
                                fileObject.write(str(user) + '::' + str(choose) + '::' + str(rank[choose]) + '\n')

if __name__=='__main__':
    '''
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')
    itemList = FO.readItemList('data/movies.txt')  # 获取物品列表
    predictSystem(trainDataSet,testDataSet,itemList,1,)
    testSysytem()
    '''
    '''
    BestParameter = chooseBestParameter()
    for key, value in BestParameter.items():
        print(key, value)
    with open('best_result_itemCF.txt','w') as fileObject:
        for key, values in BestParameter.items():
            for value in values:
                fileObject.write(str(value)+',')
            fileObject.write(str(key)+'\n')
    '''
    '''
    F_list = [2, 5, 10]
    step_list = [5, 10, 20, 30]
    LearnRating_list = [0.01, 0.05, 0.1]
    penalty_list = [0.01, 0.05, 0.1, 0.2]
    initializeSystem(F_list, step_list,LearnRating_list,penalty_list)
    '''

    '''
    F_list = [10]
    step_list = [10]
    LearnRating_list = [0.01]
    penalty_list = [0.1]
    fillMatrix(F_list,step_list,LearnRating_list,penalty_list,N = 50)
    '''

    F_list = [5]
    step_list = [10]
    LearnRating_list = [0.05]
    penalty_list = [0.05]
    testSystem(F_list,step_list,LearnRating_list,penalty_list)

