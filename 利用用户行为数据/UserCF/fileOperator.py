import random

# 将数据集写入到文件之中
def writeDataSet(filename,dataSet):
    with open(filename, 'w') as fileObject:
        for user_item, rating in dataSet.items():
            user = user_item[0]
            item = user_item[1]
            fileObject.write(str(user) + '::')
            fileObject.write(str(item) + '::')
            fileObject.write(str(rating) + '\n')

# 读取数据
def readDataSet(filename):
    dataSet = dict()
    with open(filename) as fileObject:
        for line in fileObject.readlines():
            arr = list(map(int,line.split('::')))
            dataSet[arr[0], arr[1]] = arr[2]
    return dataSet

# 读取用户列表
def readUserList(filename):
    userList = set()
    with open(filename) as fileObject:
        for line in fileObject.readlines():
            arr = line.split('::')
            userList.add(int(arr[0]))
    return userList

# 读取物品列表
def readItemList(filename):
    itemList = set()
    with open(filename) as fileObject:
        for line in fileObject.readlines():
            arr = line.split('::')
            itemList.add(int(arr[0]))
    return itemList

# 分割数据
def splitData(data, M, k, seed):
    test = dict()  # 记录测试机
    train = dict()  # 记录训练集
    random.seed(seed)  # 设置随机数种子
    #print(data)
    for user_item, rating in data.items():
        if random.randint(0, M) == k:
            #print([user, item])
            test[user_item[0], user_item[1]] = rating
        else:
            train[user_item[0], user_item[1]] = rating
    return train, test


# 读取原始数据
def readFromFile(filename, type):
    if type == 'ratings':
        data = dict()
        with open(filename) as fileObject:
            for line in fileObject.readlines():
                arr = list(map(int,line.split('::')))
                data[arr[0], arr[1]] = arr[2]
                #print((arr[0], arr[1]), ':', arr[2])
        return data
    else:
        print('No such dataSet')
        return None

# 读取该用户与其他用户的相似度
def readUserSimilarity(user,type):
    if type == 'implicit':
        filename = 'UserCF/UserSimilarity/implicit/' + str(user) + '.txt'
    elif type == 'cosSim':
        filename = 'UserCF/UserSimilarity/explicit/cosSim/' + str(user) + '.txt'
    elif type == 'ecludSim':
        filename = 'UserCF/UserSimilarity/explicit/ecludSim/' + str(user) + '.txt'
    else :
        print('no such type of file')
        return None

    W = dict()
    with open(filename) as fileObject:
        for line in fileObject.readlines():
            arr = line.split('::')
            W[int(arr[0]), int(arr[1])] = float(arr[2])
    return W

