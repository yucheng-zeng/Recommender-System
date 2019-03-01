from numpy import *
from numpy import linalg as la
import fileOperator as FO
import operator

# 计算余弦相似度
def cosSim(inA,inB):
    num = dot(inA,inB.T)  # 分子
    denom = la.norm(inA)*la.norm(inB)  # 分母, la.norm() 是计算向量的范数
    return 0.5+0.5*(num/denom)

# 计算欧氏距离, 归化到0~1
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

# 规范化相似度
def normalized(W):
    valueList = [item for user, item in W.items()]
    print(sum(valueList))
    print(valueList)
    maxValue = max(valueList)
    print(maxValue)
    valueList = [item/maxValue for item in valueList]
    print(sum(valueList))
    print(valueList)
    for user,value in W.items():
        W[user[0], user[1]] = value/maxValue


# 计算物品之间的隐式相似度
def ItemSimilarity_implicit(users_item, target_item):
    # calculate co-rated users between items
    C = dict()
    N = dict()
    for user, items in users_item.items():
        for item_i in items:
            N[item_i] = N.get(item_i, 0) + 1
            if target_item == item_i:
                for item_j in items:
                    if item_i == item_j:
                        continue
                    C[item_i, item_j] = C.get((item_i, item_j),0) + 1/math.log(1+len(items))

    # calculate finial similarity matrix W
    W = dict()  #
    for related_items, cij in C.items():
        W[related_items[0], related_items[1]] = cij / math.sqrt(N[related_items[0]] * N[related_items[1]])
    return W


# 显式计算用户user 与 所有用户之间的相似度
def ItemSimilarity_explicit(trainDataSet, users_item, itemList, item, simMeas=cosSim):
    # 计算两个用户之间的相似度
    W = dict()  # 记录用户之间的相似程度
    for otherItem in itemList:  #
        if otherItem != item:
            itemInA = list()  # 记录用户A的物品向量
            itemInB = list()  # 记录用户B的物品向量
            for user, itemContant in users_item.items():
                if otherItem in itemContant and item in itemContant:
                    #print(user,',',item,',',trainDataSet[user,item])
                    #print(otherUser,',',item,',',trainDataSet[otherUser,item])
                    itemInA.append(trainDataSet[user,item])
                    itemInB.append(trainDataSet[user,otherItem])
            #print(itemInA)
            #print(itemInB)
            if len(itemInA) != 0:
                value = simMeas(array(itemInA),array(itemInB))  # 计算两个向量之间的相似度,也就是用户之间的相似度
            else:
                if simMeas.__name__ == 'cosSim':
                    value = 0.69
                else:
                    value = 0.2
            W[item,otherItem] = value
            #print(user,',',otherUser,',',value)
    return W


# 初始化系统
def initializeSystem(trainRating=20,testRating=1,seed=0):
    # 重新划分训练集 以及 测试集
    data = FO.readFromFile('data/ratings.txt', 'ratings')
    trainDataSet, testDataSet = FO.splitData(data, trainRating, testRating, seed)
    print(trainDataSet)
    # 将 训练集 与 测试集 写入到文件之中
    FO.writeDataSet('ItemCF/trainDataSet.txt', trainDataSet)
    FO.writeDataSet('ItemCF/testDataSet.txt', testDataSet)
    # 重新从新计算用户之间的相似度
    calItemSimilarity(trainDataSet, 'implicit')  # 计算隐式相似度
    calItemSimilarity(trainDataSet, 'explicit',cosSim)  # 以余弦相似度计算显式相似度
    calItemSimilarity(trainDataSet, 'explicit',ecludSim)  # 以欧氏距离计算显式相似度


# 计算用户之间的相似度
def calItemSimilarity(trainDataSet, type='implicit', simMeas=cosSim):
    '''
    :param trainDataSet: 训练数据集
    :param type: 以何种方式计算相似度
    :param simMeas: 计算相似的方法
    :return: None
    最后将用户之间的相似度写到文件里面
    '''

    # 建立 用户 到 物品 的倒排表
    users_item = dict()
    for user, item in trainDataSet:
        if user not in users_item:
            users_item[user] = set()
        users_item[user].add(item)
    for user, item in users_item.items():
        print(user, ':', item)

    # for item, user in item_users.items():
    #    print(item, ':', user)

    # 读取用户列表
    itemList = FO.readItemList('data/movies.txt')

    # 遍历每一个用户, 计算用户之间的相似度
    for item in itemList:
        print(item)
        if type == 'implicit':  # 判读计算相似度的方式, 这里隐式计算
            W = ItemSimilarity_implicit(users_item, item)
            filename = 'ItemCF/ItemSimilarity/implicit/' + str(item) + '.txt'
        elif type == 'explicit': # 判读计算相似度的方式, 这里显式方式
            W = ItemSimilarity_explicit(trainDataSet, users_item, itemList, item,simMeas)
            filename ='ItemCF/ItemSimilarity/explicit/'+ str(simMeas.__name__)+'/'+str(item) + '.txt'

        # 将结果写入文件之中
        with open(filename, 'w') as fileObject:
            for users, values in W.items():
                for u in users:
                    fileObject.write(str(str(u) + '::'))
                fileObject.write(str(values) + '\n')


# 测试系统
def testSystem(trainDataSet, testDataSet, K=20, N=30, type='implicit'):
    testUserList = set()  # 记录测试数据集里面的用户列表
    for user_item, value in testDataSet.items():
        testUserList.add(user_item[0])

    itemList = FO.readItemList('data/movies.txt')  # 获取物品列表

    recall = 0  # 计算总的召回率
    precision = 0  # 计算总的精确度
    coverage_item = set()  # 计算覆盖的物品

    userLove = dict()
    for user in testUserList:  # 遍历所有测试用户
        #userLove[user] = [(user_item[1],value) for user_item, value in trainDataSet.items() if user_item[0] == user]
        userLove[user] = set([user_item[1] for user_item, value in trainDataSet.items() if user_item[0] == user])
        rank = Recommend(trainDataSet, itemList, userLove[user], user, K, type)
        print(rank)
        if N<len(rank):
            top = N
        else:
            top = len(rank)
        topRank = sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[:top]  # 获取排名前N个的推荐物品
        topItemRank = [item_rating[0] for item_rating in topRank]  # 获取排名前N个的推荐物品
        User_item_InTest = [user_item[1] for user_item, rating in testDataSet.items() if user_item[0] == user]  # 获取用户在测试集里面喜欢的物品
        #print(User_item_InTest)
        #print(topRank)
        #print(topItemRank)
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
def Recommend(trainDataSet, itemList, userLove, user, K, type):
    rank = dict()
    for item in itemList:  # 遍历物品列表
        if item not in userLove:  # 若果该用户对该物品没有评分, 则进行预测
            W = FO.readItemSimilarity(item, type)  # 获取当前用户与其他用户的相似度
            topItem = dict(sorted(W.items(), key=operator.itemgetter(1), reverse=True)[:K])
            topSimItem = set([items[1] for items, value in topItem.items()])
            topLoveItem = userLove & topSimItem
            if len(topLoveItem) == 0:
                continue
            #print(topLoveItem)
            for LoveItem in topLoveItem:
                wji = W[item, LoveItem]
                rank[item] = rank.get(item, 0) + wji * trainDataSet[user, LoveItem]
    return rank


# 计算召回率
def Recall(User_item_InTest, rank):
    hit = 0  # 记录推荐列表中有多少个是准确推送
    all = len(User_item_InTest)  # 用户在测试集上喜欢的物品
    for item in User_item_InTest:
        if item in rank:
            hit += 1  # 猜中加1
    return hit / (all * 1.0)

# 计算准确女
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


def predictSystem(trainDataSet, testDataSet, user, itemList, K=20, N=30, type='implicit'):
    '''
    :param trainDataSet: 训练数据集
    :param testDataSet: 测试数据集
    :param user: 当前用户
    :param itemList: 物品列表
    :param K: 前K个相似度最高的用户
    :param N: 选取前N个评分最高的物品
    :return: 返回N个评分最高的物品
    '''
    dataSet = {**trainDataSet, **testDataSet}
    userLove = set([user_item[1] for user_item, value in dataSet.items() if user_item[0] == user])
    rank = dict()
    for item in itemList:  # 遍历物品列表
        if item not in userLove:  # 若果该用户对该物品没有评分, 则进行预测
            W = FO.readItemSimilarity(item, type)  # 获取当前用户与其他用户的相似度
            topItem = dict(sorted(W.items(), key=operator.itemgetter(1), reverse=True)[:K])
            topSimItem = set([items[1] for items, value in topItem.items()])
            topLoveItem = userLove & topSimItem
            if len(topLoveItem) == 0:
                continue
            # print(topLoveItem)
            for LoveItem in topLoveItem:
                wji = W[item, LoveItem]
                rui = dataSet[user, LoveItem]
                rank[item] = rank.get(item, 0) + wji * rui
    return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])


if __name__ == '__main__':

    trainDataSet = FO.readDataSet('ItemCF/trainDataSet.txt')
    testDataSet = FO.readDataSet('ItemCF/testDataSet.txt')
    '''
    calItemSimilarity(trainDataSet, 'implicit')  # 计算隐式相似度
    calItemSimilarity(trainDataSet, 'explicit', cosSim)  # 以余弦相似度计算显式相似度
    calItemSimilarity(trainDataSet, 'explicit', ecludSim)  # 以欧氏距离计算显式相似度
    '''
    itemList = FO.readItemList('data/movies.txt')
    for i in range(1,11):
        rank = predictSystem(trainDataSet, testDataSet, i, itemList)
        print(rank)


