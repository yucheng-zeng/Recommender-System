import math
import random
import operator
import fileOperator as FO
from numpy import *
import datetime


# 测试系统
def testSystem(trainDataSet, testDataSet, K=20, N=30, type='implicit'):
    testUserList = set()  # 记录测试数据集里面的用户列表
    for user_item, value in testDataSet.items():
        testUserList.add(user_item[0])

    itemList = FO.readItemList('data/movies.txt')  # 获取物品列表

    recall = 0  # 计算总的召回率
    precision = 0  # 计算总的精确度
    coverage_item = set()  # 计算覆盖的物品
    for user in testUserList:  # 遍历所有测试用户
        W = FO.readUserSimilarity(user,type)  # 获取当前用户与其他用户的相似度
        rank = Recommend(trainDataSet,itemList,user,W, K)  # 计算推荐物品排名
        if N<len(rank):
            top = N
        else:
            top = len(rank)
        topRank = sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[:top]  # 获取排名前N个的推荐物品
        topItemRank = [item_rating[0] for item_rating in topRank]  # 获取排名前N个的推荐物品
        User_item_InTest = [user_item[1] for user_item, rating in testDataSet.items() if user_item[0] == user]  # 获取用户在测试集里面喜欢的物品
        # 计算 准确率 以及 召回率
        oneRecall = Recall(User_item_InTest, topItemRank)
        onePrecision = Precision(User_item_InTest, topItemRank)
        recall += oneRecall
        precision += onePrecision
        print('recall=%.4f,precision=%.4f'%(oneRecall,onePrecision))
        # 计算覆盖率
        Coverage(coverage_item, topItemRank)

    coverage = float(len(coverage_item)) / len(itemList)
    popularity = Popularity(trainDataSet, coverage_item)
    recall = recall/float(len(testUserList))
    precision = precision/float(len(testUserList))
    print('total recall=%.4f,precision=%.4f,coverage=%.4f,popularity=%.4f'%\
          (recall,precision,coverage,popularity))

    return recall, precision, coverage, popularity


# 给用户未打分的物品进行预测评分
def Recommend(trainDataSet, itemList, user, W, K):
    rank = dict()  # 记录用户物品 以及 评分
    User_item_InTrain = [user_item[1] for user_item, rating in trainDataSet.items() if user_item[0]==user]  # 获取用户在训练集里面的物品
    #print(User_item_InTrain)
    topUser = dict(sorted(W.items(), key=operator.itemgetter(1), reverse=True)[:K])  # 选取与用户相似度最高的前K的用户
    #print(topUser)
    for item in itemList:  # 遍历物品列表
        if item not in User_item_InTrain:  # 若果该用户对该物品没有评分, 则进行预测
            for users, value in topUser.items():
                wuv = value  # 获取用户之间的相似度
                # 获取其他用户对该物品的打分
                if (users[1],item) in trainDataSet.keys():
                    #print('i am here')
                    rvi = trainDataSet[users[1], item]
                else:
                    rvi = 0
                rank[item] = rank.get(item, 0) + wuv * rvi
                #if rvi != 0:
                #    print(item,rank[item],wuv,rvi)
    return rank



# 计算召回率
def Recall(User_item_InTest, rank):
    hit = 0  # 记录推荐列表中有多少个是准确推送
    all = len(User_item_InTest)  # 用户在测试集上喜欢的物品
    for item in User_item_InTest:
        if item in rank:
            hit += 1  # 猜中加1
    return hit / (all * 1.0)

# 计算准确率
def Precision(User_item_InTest, rank):
    hit = 0  # 记录推荐列表中有多少个是准确推送
    all = len(rank)  # 用户在测试集上喜欢的物品
    for item in User_item_InTest:
        if item in rank:
            hit += 1  # 猜中加1
    return hit / (all * 1.0)

# 计算覆盖率
def Coverage(coverage_item,topItemRank):
    for item in topItemRank:
        coverage_item.add(item)

# 计算物品流行度
def Popularity(trainDataSet, coverage_item):
    item_popularity = dict()
    for user_item, rating in trainDataSet.items():
            item_popularity[user_item[1]] = item_popularity.get(user_item[1], 0) + 1
    ret = 0
    n = 0
    for item in coverage_item:
        ret += math.log(1 + item_popularity[item])
        n += 1
    ret /= n * 1.0
    return ret


# 预测系统
def predictSystem(trainDataSet, testDataSet, user, itemList, K=20, N=30):
    '''
    :param trainDataSet: 训练数据集
    :param testDataSet: 测试数据集
    :param user: 当前用户
    :param itemList: 物品列表
    :param K: 前K个相似度最高的用户
    :param N: 选取前N个评分最高的物品
    :return: 返回N个评分最高的物品
    '''
    W = FO.readUserSimilarity(user, 'single')  # 获取当前用户与其他用户的相似度
    topSimilarity = dict(sorted(W.items(), key=operator.itemgetter(1), reverse=True)[:K])
    #print(topSimilarity)
    rank = dict()
    for item in itemList:  # 遍历物品列表
        if (user,item) not in trainDataSet and (user,item) not in testDataSet:  # 判断该物品用户是否已经对其进行了评分
            for users, similarity in topSimilarity.items():
                wuv = similarity
                # 获取其他用户对该物品的频分
                if (users[1], item) in trainDataSet.keys():
                    rvi = trainDataSet[users[1], item]
                elif (users[1], item) in testDataSet.keys():
                    rvi = testDataSet[users[1], item]
                else:
                    rvi = 0
                rank[item] = rank.get(item, 0) + wuv * rvi  # 计算该用户对该物品的评分
    #print(rank)
    return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])


if __name__ == '__main__':
    '''
    trainDataSet = FO.readDataSet('UserCF/trainDataSet.txt')
    testDataSet = FO.readDataSet('UserCF/testDataSet.txt')
    KList = [5, 10, 20, 30, 40]
    NList = [10,15, 20, 30]
    for k in KList:
        for n in NList:
            recall, precision, coverage, popularity = testSystem(trainDataSet,testDataSet,K=k,N=n)
            with open('UserCF/Result/result.txt','a') as fileObject:
                fileObject.write(str(k)+','+str(n)+','+str(recall)+','+str(precision)+','+str(coverage)+','+str(popularity)+'\n')
    '''

    '''
    trainDataSet = FO.readDataSet('UserCF/trainDataSet.txt')
    testDataSet = FO.readDataSet('UserCF/testDataSet.txt')
    calUserSimilarity(trainDataSet,'explicit',cosSim)
    '''
    '''
    trainDataSet = FO.readDataSet('UserCF/trainDataSet.txt')
    testDataSet = FO.readDataSet('UserCF/testDataSet.txt')
    itemList = FO.readItemList('data/movies.txt')
    rank = predictSystem(trainDataSet, testDataSet, 1, itemList)
    print(rank)
    '''

    timeList = []  # 记录程序运行时间
    timeList.append(datetime.datetime.now())
    # 加载数据
    trainDataSet = FO.readDataSet('UserCF/trainDataSet.txt')
    testDataSet = FO.readDataSet('UserCF/testDataSet.txt')
    KList = [10, 20, 30, 40]
    NList = [10, 20, 30, 40]
    type_list = ['implicit','cosSim','ecludSim']
    for type in type_list:
        for k in KList:
            for n in NList:
                recall, precision, coverage, popularity = testSystem(trainDataSet,testDataSet,K=k,N=n,type=type)
                with open('UserCF/Result/result_'+type+'.txt','a') as fileObject:
                    fileObject.write(str(k)+','+str(n)+','+str(recall)+','+str(precision)+','+str(coverage)+','+str(popularity)+'\n')
    timeList.append(datetime.datetime.now())
    print(timeList)




    '''
    initializeSystem(10,1,2)

    timeList.append(datetime.datetime.now())

    trainDataSet = FO.readDataSet('UserCF/trainDataSet.txt')
    testDataSet = FO.readDataSet('UserCF/testDataSet.txt')
    KList = [5, 10, 20, 30, 40, 50]
    NList = [10, 15, 20, 30]
    type_list = ['implicit', 'cosSim', 'ecludSim']
    for type in type_list:
        for k in KList:
            for n in NList:
                recall, precision, coverage, popularity = testSystem(trainDataSet, testDataSet, K=k, N=n, type=type)
                with open('UserCF/Result/result_' + type + '.txt', 'a') as fileObject:
                    fileObject.write(str(k) + ',' + str(n) + ',' + str(recall) + ',' + str(precision) + ',' + str(
                        coverage) + ',' + str(popularity) + '\n')

    timeList.append(datetime.datetime.now())

    '''
