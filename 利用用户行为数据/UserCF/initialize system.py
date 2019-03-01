import fileOperator as FO
from numpy import linalg as la
from numpy import *

# 计算余弦相似度
def cosSim(inA, inB):
    num = dot(inA,inB)  # 分子
    denom = la.norm(inA)*la.norm(inB)  # 分母, la.norm() 是计算向量的范数
    return 0.5+0.5*(num/denom)  # 规范化到0~1

# 计算欧氏距离, 归化到0~1
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))  # 规范化到0~1


# 计算用户之间的相似度
def calUserSimilarity(trainDataSet, type='implicit', simMeas=cosSim):
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

    # 遍历每一个用户, 计算用户之间的相似度
    for user in userList:
        print(user)
        if type == 'implicit':  # 判读计算相似度的方式, 这里隐式计算
            W = UserSimilarity_implicit(userList, user)
            filename = 'UserCF/UserSimilarity/implicit/' + str(user) + '.txt'
        elif type == 'explicit': # 判读计算相似度的方式, 这里显式方式
            W = UserSimilarity_explicit(trainDataSet, item_users, userList, user,simMeas)
            filename ='UserCF/UserSimilarity/explicit/'+ str(simMeas.__name__)+'/'+str(user) + '.txt'
        # 将结果写入文件之中
        with open(filename, 'w') as fileObject:
            for users, values in W.items():
                for u in users:
                    fileObject.write(str(str(u) + '::'))
                fileObject.write(str(values) + '\n')


# 隐式计算用户user 与 所有用户之间的相似度
def UserSimilarity_implicit(item_users, user):
    # 计算两个用户之间的相似度
    C = dict()  # 计算两用户之间共同喜欢物品的交集
    N = dict()  # 计算某一位用户喜欢的物品
    for i, users in item_users.items():  # 遍历物品 到 用户 的倒排表
        for u in users:
            N[u] = N.get(u, 0) + 1
            if u == user:
                for v in users:
                    if u == v:
                        continue
                    C[u, v] = C.get((u, v), 0) + 1 / math.log(1 + len(users))

    #for users, values in C.items():
    #    print(users,':',values)

    # 若果用户相似列表为空则返回None
    if len(C) == 0:
        return None

    # 计算相似程度
    W = dict()  # 记录用户之间的相似程度
    for related_users, cuv in C.items():
            W[related_users[0], related_users[1]] = cuv / math.sqrt(N[related_users[0]] * N[related_users[1]])
    return W


# 显式计算用户user 与 所有用户之间的相似度
def UserSimilarity_explicit(trainDataSet, item_users, userList, user,simMeas=cosSim):
    # 计算两个用户之间的相似度
    W = dict()  # 记录用户之间的相似程度
    for otherUser in userList:  #
        if otherUser != user:
            userInA = list()  # 记录用户A的物品向量
            userInB = list()  # 记录用户B的物品向量
            for item, usersContant in item_users.items():
                if otherUser in usersContant and user in usersContant:
                    #print(user,',',item,',',trainDataSet[user,item])
                    #print(otherUser,',',item,',',trainDataSet[otherUser,item])
                    userInA.append(trainDataSet[user,item]-1)
                    userInB.append(trainDataSet[otherUser,item]-1)
            if len(userInA) != 0:
                value = simMeas(array(userInA),array(userInB))  # 计算两个向量之间的相似度,也就是用户之间的相似度
            else:
                value = 0
            W[user,otherUser] = value
            #print(user,',',otherUser,',',value)
    return W



# 初始化系统
def initializeSystem(trainRating=20,testRating=1,seed=0):
    # 重新划分训练集 以及 测试集
    data = FO.readFromFile('data/ratings.txt', 'ratings')
    trainDataSet, testDataSet = FO.splitData(data, trainRating, testRating, seed)
    print(trainDataSet)
    # 将 训练集 与 测试集 写入到文件之中
    FO.writeDataSet('UserCF/trainDataSet.txt', trainDataSet)
    FO.writeDataSet('UserCF/testDataSet.txt', testDataSet)
    # 重新从新计算用户之间的相似度
    calUserSimilarity(trainDataSet,'implicit')  # 计算隐式相似度
    calUserSimilarity(trainDataSet,'explicit',cosSim)  # 以余弦相似度计算显式相似度
    calUserSimilarity(trainDataSet,'explicit',ecludSim)  # 以欧氏距离计算显式相似度

if __name__ == '__main__':
    initializeSystem()
