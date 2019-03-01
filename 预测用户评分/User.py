import math
import random
import operator
import fileOperator as FO
# 计算用户之间的相似度


def calUserSimilarity(trainDataSet):
    '''
    :param trainDataSet: 训练数据集
    :param type: 以何种方式计算相似度
    :param simMeas: 计算相似的方法
    :return: None
    最后将用户之间的相似度写到文件里面
    '''
    # 建立 物品 到 用户 的倒排表
    item_users = dict()
    for user_item, rating in trainDataSet.items():
        user = user_item[0]
        item = user_item[1]
        if item not in item_users:
            item_users[item] = set()
        item_users[item].add(user)

    # for item, user in item_users.items():
    #    print(item, ':', user)

    # 读取用户列表
    userList = FO.readUserList('data/users.txt')
    UserRatingMean = FO.ReadRatingMean('UserCF/UserMean/add_trainDataSetRating_mean.txt')
    # 遍历每一个用户, 计算用户之间的相似度
    for user in userList:
        print(user)
        W = UserSimilarity(trainDataSet,item_users,userList,UserRatingMean,user)
        filename = 'UserCF/Similarity_add/'+str(user)+'.txt'
        FO.WirteSimilarty(filename, W)


# 隐式计算用户user 与 所有用户之间的相似度
def UserSimilarity(trainDataSet, item_users, userList, userRatingMean, user):
    '''

    UserMean = userRatingMean[user]
    W = dict()
    for otherUser in userList:
        molecular = 0
        denominator1 = 0
        denominator2 = 0
        if otherUser != user:
            otherUserMean = userRatingMean[otherUser]
            for item, usersContant in item_users.items():
                if otherUser in usersContant and user in usersContant:
                    raing1 = trainDataSet[user, item]
                    rating2 = trainDataSet[otherUser, item]
                    molecular += (raing1 - UserMean)*(rating2 - otherUserMean)
                    denominator1 += (raing1 - UserMean)*(raing1 - UserMean)
                    denominator2 += (rating2 - otherUserMean)*(rating2 - otherUserMean)
            W[user, otherUser] = molecular / math.sqrt((denominator2+1)*(denominator1+1))
    return W
    '''

    molecular = dict()
    denominator1 = dict()
    denominator2 = dict()
    W = dict()
    UserMean = userRatingMean[user]
    for item, userContent in item_users.items():
        if user in userContent:
            for otherUser in userContent:
                if otherUser != user:
                    otherUserMean = userRatingMean[otherUser]
                    rui = trainDataSet[user, item]
                    ruj = trainDataSet[otherUser, item]
                    molecular[otherUser] = molecular.get(otherUser, 0) + (rui - UserMean)*(ruj - otherUserMean)
                    denominator1[otherUser] = denominator1.get(otherUser, 0) + (rui - UserMean)*(rui - UserMean)
                    denominator2[otherUser] = denominator2.get(otherUser, 0) + (ruj - otherUserMean)*(ruj - otherUserMean)
    for otherUser in userList:
        if otherUser == user:
            continue
        if otherUser not in molecular.keys():
            W[user, otherUser] = 0
        else:
            W[user, otherUser] = molecular[otherUser] / math.sqrt((denominator2[otherUser]+0.01)*(denominator1[otherUser]+0.01))
    return W



# 初始化系统
def initializeSystem(trainRating=20,testRating=1,seed=0):
    # 重新划分训练集 以及 测试集
    data = FO.readFromFile('data/ratings.txt', 'ratings')
    trainDataSet, testDataSet = FO.splitData(data, trainRating, testRating, seed)
    #print(trainDataSet)
    # 将 训练集 与 测试集 写入到文件之中
    FO.writeDataSet('UserCF/trainDataSet.txt', trainDataSet)
    FO.writeDataSet('UserCF/testDataSet.txt', testDataSet)
    # 重新从新计算用户之间的相似度
    calUserSimilarity(trainDataSet)


# 测试系统
def testSystem(trainDataSet, testDataSet, K=30):
    error = 0
    userMeanDict = FO.ReadRatingMean('UserCF/UserMean/add_trainDataSetRating_mean.txt')
    for user_item, rating in testDataSet.items():
        predict = Recommend(trainDataSet, user_item, userMeanDict, K)
        print('(',user_item[0],user_item[1],'):',predict)
        error += (predict-rating)*(predict-rating)
    RMSE = math.sqrt(error/len(testDataSet))
    print('RMSE=%0.3f'%(RMSE))




# 给用户未打分的物品进行预测评分
def Recommend(trainDataSet,user_item, userMeanDict, K):
    user = user_item[0]
    item = user_item[1]
    #print(item)
    W = FO.readSimilarity('UserCF/Similarity_add/'+str(user)+'.txt')
    topSimilarity = dict(sorted(W.items(), key=operator.itemgetter(1), reverse=True)[:K])
    #print(topSimilarity)
    molecular = 0
    denominator = 0
    for users, similarity in topSimilarity.items():
        otherUser = users[1]
        if (otherUser,item) not in trainDataSet.keys():
            continue
        otherUserMean = userMeanDict[otherUser]
        wuv = similarity  # 获取用户之间的相似度
        rvi = trainDataSet[otherUser, item]  # 获取其他用户对该物品的打分
        molecular += wuv*(rvi-otherUserMean)
        denominator += math.fabs(wuv)
    rating = userMeanDict[user] + molecular/(denominator+0.01)
    if rating > 5:
        rating = 5
    elif rating < 1:
        rating = 1
    return rating


# 预测系统
def predictSystem(trainDataSet, testDataSet, itemList, userMeanDict, user, K=20, N=30):
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
    W = FO.readSimilarity(user)  # 获取当前用户与其他用户的相似度
    topSimilarity = dict(sorted(W.items(), key=operator.itemgetter(1), reverse=True)[:K])
    rank = dict()
    for item in itemList:  # 遍历物品列表
        if (user,item) not in dataSet:  # 判断该物品用户是否已经对其进行了评分
            molecular = 0
            denominator = 0
            for users, similarity in topSimilarity.items():
                otherUser = users[1]
                if (otherUser,item) not in dataSet.keys():
                    continue
                otherUserMean = userMeanDict[otherUser]
                wuv = similarity  # 获取用户之间的相似度
                rvi = dataSet[otherUser, item]  # 获取其他用户对该物品的打分
                molecular += wuv * (rvi - otherUserMean)
                denominator += math.fabs(wuv)

            rank[item] = userMeanDict[user] + molecular / (denominator + 0.01)  # 计算该用户对该物品的评分
            if rank[item] > 5:
                rank[item] = 5.00
    return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])


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
    #addDataSet = FO.readDataSet('SVD++/Matrix/5-5-0.01-0.1/new_ratings.txt')
    trainDataSet = {**trainDataSet, **testDataSet}
    userMeanDict = calUserRatingMean(trainDataSet)
    FO.WirteRatingMean('UserCF/UserMean/DataSetUserRating_mean.txt', userMeanDict)
    #calUserSimilarity(trainDataSet)
    '''
    userList = FO.readUserList('data/users.txt')
    
    userMeanDict =calUserRatingMean(trainDataSet)
    FO.WirteUserRatingMean('trainDataSetUserRatingMean.txt',userMeanDict)

    data = FO.readFromFile('data/ratings.txt', 'ratings')
    allUserMeanDict = calUserRatingMean(data)
    FO.WirteUserRatingMean('dataSetUserRatingMean.txt',allUserMeanDict)
    '''

    '''
    userMeanDict = FO.ReadUserRatingMean('UserCF/UserMean/trainDataSetUserRatingMean.txt')
    itemList = FO.readItemList('data/movies.txt')  # 获取物品列表
    userList = FO.readUserList('data/users.txt')
    for user in userList:
        if user > 150:
            break
        W = FO.readUserSimilarity(user)
        rank = Recommend(trainDataSet,itemList,userMeanDict,user,W, 20)
        print(rank)
    '''
    '''
    itemList = FO.readItemList('data/movies.txt')  # 获取物品列表
    userMeanDict = FO.ReadUserRatingMean('UserCF/UserMean/trainDataSetUserRatingMean.txt')
    for i in range(1, 100):
        rank = predictSystem(trainDataSet,testDataSet,itemList,userMeanDict,i)
        print(rank)
    '''

    #testSystem(trainDataSet,testDataSet,K=50)
