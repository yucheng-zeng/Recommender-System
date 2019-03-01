from numpy import *
from numpy import linalg as la
import fileOperator as FO
import operator


def ItemSimilarity(trainDataSet,user_items,itemList, UserRatingMean, target_item):
    molecular = dict()
    denominator1 = dict()
    denominator2 = dict()

    W = dict()
    for user, itemContent in user_items.items():
        userMean = UserRatingMean[user]
        if target_item in itemContent:
            for item in itemContent:
                if item != target_item:
                    rui = trainDataSet[user, target_item]
                    ruj = trainDataSet[user, item]
                    molecular[item] = molecular.get(item, 0) + (rui - userMean)*(ruj-userMean)
                    denominator1[item] = denominator1.get(item, 0) + (rui-userMean)*(rui-userMean)
                    denominator2[item] = denominator2.get(item, 0) + (ruj-userMean)*(ruj-userMean)
    for item in itemList:
        if item != target_item:
            if item not in molecular.keys():
                W[target_item, item] = 0
            else:
                W[target_item, item] = molecular[item]/math.sqrt((denominator1[item]+0.01)*(denominator2[item]+0.01))
    return W
    '''
    W = dict()
    for item in itemList:
        if item != target_item:
            molecular = 0
            denominator1 = 0
            denominator2 = 0
            for user, itemContent in user_items.items():
                if target_item in itemContent and item in itemContent:
                    userMean = UserRatingMean[user]
                    rui = trainDataSet[user,target_item]
                    ruj = trainDataSet[user,item]
                    molecular += (rui-userMean)*(ruj-userMean)
                    denominator1 += (rui-userMean)*(rui-userMean)
                    denominator2 += (ruj-userMean)*(ruj-userMean)
            W[target_item, item] = molecular / math.sqrt((denominator1+0.01)*(denominator2+0.01))
    return W
    '''
    
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



# 计算用户之间的相似度
def calItemSimilarity(trainDataSet):
    '''
    :param trainDataSet: 训练数据集
    :param type: 以何种方式计算相似度
    :param simMeas: 计算相似的方法
    :return: None
    最后将用户之间的相似度写到文件里面
    '''

    # 建立 用户 到 物品 的倒排表
    user_items = dict()
    for user, item in trainDataSet:
        if user not in user_items:
            user_items[user] = set()
        user_items[user].add(item)

    #for user, item in user_items.items():
    #    print(user, ':', item)

    # for item, user in item_users.items():
    #    print(item, ':', user)

    # 读取用户列表
    itemList = FO.readItemList('data/movies.txt')
    #ItemRatingMean = FO.ReadRatingMean('ItemCF/ItemMean/add_trainDataSet_mean.txt')
    UserRatingMean = FO.ReadRatingMean('ItemCF/UserMean/trainDataSet_mean.txt')
    # 计算用户之间的相似度
    for item in itemList:
        W = ItemSimilarity(trainDataSet, user_items, itemList, UserRatingMean, item)
        print(item)
        filename = 'ItemCF/Similarity/'+str(item)+'.txt'
        FO.WirteSimilarty(filename,W)

# 测试系统
def testSystem(trainDataSet, testDataSet, K=20):
    itemMean = FO.ReadRatingMean('ItemCF/ItemMean/trainDataSetRatingMean.txt')
    error = 0
    for user_item, rating in testDataSet.items():
        predict = Recommend(trainDataSet, user_item, itemMean, K)
        print(user_item, predict)
        error += (predict - rating)*(predict - rating)
    RMSE = math.sqrt(error / len(testDataSet))
    print('RMSE=%0.3f' % (RMSE))


# 给用户未打分的物品进行预测评分
def Recommend(trainDataSet,user_item, itemMean, K):
    user = user_item[0]
    item = user_item[1]
    W = FO.readSimilarity('ItemCF/Similarity/'+str(item)+'.txt')
    topSimilarity = dict(sorted(W.items(), key=operator.itemgetter(1), reverse=True)[:K])
    molecular = 0
    denominator = 0
    for items, similarity in topSimilarity.items():
        otherItem = items[1]
        if (user, otherItem) not in trainDataSet.keys():
            continue
        otherItemMean = itemMean[otherItem]
        ruj = trainDataSet[user, otherItem]
        wij = similarity
        molecular += wij*(ruj - otherItemMean)
        denominator += math.fabs(wij)
    rating = itemMean[item] + molecular/(denominator+0.01)
    if rating > 5:
        rating = 5
    elif rating < 1:
        rating = 1
    return rating


def predictSystem(trainDataSet, testDataSet,itemMean,user, itemList, K=20, N=30):
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
    userRated = set([user_item[1] for user_item, value in dataSet.items() if user_item[0] == user])
    rank = dict()
    UnRatedList = itemList - userRated
    for item in UnRatedList:  # 遍历物品列表
        molecular = 0
        denominator = 0
        W = FO.readSimilarity('ItemCF/Similarity/'+str(item)+'.txt')  # 获取当前用户与其他用户的相似度
        topItem = dict(sorted(W.items(), key=operator.itemgetter(1), reverse=True)[:K])  # 获取前K个与该用户最相似的用户
        topSimItem = set([items[1] for items, value in topItem.items()]) # 获取这些用户评分过的物品
        topRatedItem = userRated & topSimItem  
        if len(topRatedItem) == 0:
            continue
        for LoveItem in topRatedItem:
            if LoveItem == item:
                continue
            wij = W[item, LoveItem]
            ruj = dataSet[user, LoveItem]
            molecular += W[item,LoveItem]*(ruj-itemMean[LoveItem])
            denominator += math.fabs(wij)
        rank[item] = itemMean[item] + (molecular/(denominator+0.01))
    return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])


def calItemRatingMean(dataSet):
    itemList = FO.readItemList('data/movies.txt')  # 获取物品列表
    itemMeanDict = dict()
    for item in itemList:
        print(item)
        itemRating = [value for user_item, value in dataSet.items() if user_item[1] == item]
        if len(itemRating) == 0:
            value = 0
        else:
            value = sum(itemRating) / len(itemRating)
        itemMeanDict[item] = value
    return itemMeanDict

def calUserRatingMean(dataSet):
    userList = FO.readUserList('data/users.txt')
    userMeanDict = dict()
    for user in userList:
        print(user)
        UserRating = [value for user_item, value in dataSet.items() if user_item[0] == user]
        userMeanDict[user] = sum(UserRating) / len(UserRating)
    return userMeanDict


if __name__ == '__main__':
    trainDataSet = FO.readDataSet('trainDataSet.txt')
    testDataSet = FO.readDataSet('testDataSet.txt')
    addDataSet = FO.readDataSet('SVD++/Matrix/10-10-0.01-0.1/new_ratings.txt')
    trainDataSet = {**trainDataSet, **addDataSet}


    itemMeanDict = calItemRatingMean(trainDataSet)
    FO.WirteRatingMean('ItemCF/ItemMean/add_trainDataSet_mean.txt', itemMeanDict)

    userMeanDict = calUserRatingMean(trainDataSet)
    FO.WirteRatingMean('ItemCF/UserMean/trainDataSet_mean.txt', userMeanDict)


    calItemSimilarity(trainDataSet)
    '''
    itemMeanDict = calItemRatingMean(trainDataSet)
    FO.WirteRatingMean('ItemCF/ItemMean/add_trainDataSet.txt', itemMeanDict)
    '''

    '''
    for key, value in itemMeanDict.items():
        print(key,value)
    '''

    '''    
    data = FO.readFromFile('data/ratings.txt', 'ratings')
    '''

    testSystem(trainDataSet, testDataSet, K=70)

    #itemMean = FO.ReadRatingMean('ItemCF/ItemMean/DataSetRatingMean.txt')
    #itemList = FO.readItemList('data/movies.txt')
    #for i in range(1, 11):
    #    rank = predictSystem(trainDataSet,testDataSet, itemMean, i, itemList)
    #    print(rank)
