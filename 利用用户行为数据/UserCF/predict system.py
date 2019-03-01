import operator
import fileOperator as FO


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
    return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])


if __name__ == '__main__':
    trainDataSet = FO.readDataSet('UserCF/trainDataSet.txt')
    testDataSet = FO.readDataSet('UserCF/testDataSet.txt')
    itemList = FO.readItemList('data/movies.txt')
    userList = [1,2,3,4]
    for user in userList:
        rank = predictSystem(trainDataSet, testDataSet, user, itemList, K=20, N=30)
        print(rank)

