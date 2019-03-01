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
def readSimilarity(filename):
    W = dict()
    with open(filename) as fileObject:
        for line in fileObject.readlines():
            arr = line.split('::')
            W[int(arr[0]), int(arr[1])] = float(arr[2])
    return W

def WirteSimilarty(filename,W):
    with open(filename,'w') as fileObject:
        for users,value in W.items():
            fileObject.write(str(users[0])+'::')
            fileObject.write(str(users[1])+'::')
            fileObject.write(str(value)+'\n')

def WirteRatingMean(filename, RatingMeanDict):
    with open(filename,'w') as fileObject:
        for user, rating in RatingMeanDict.items():
            fileObject.write(str(user)+'::')
            fileObject.write(str(rating)+'\n')

def ReadRatingMean(filename):
    RatingMean = dict()
    with open(filename) as fileObject:
        for line in fileObject.readlines():
            arr = line.split('::')
            RatingMean[int(arr[0])] = float(arr[1])
    return RatingMean

def WriteQDict(path,q,F):
    with open(path+'/q.txt','w') as fileObject:
        for key, values in q.items():
            fileObject.write(str(key) + ',')
            for f in range(F):
                if f == F-1:
                    content = str(values[f]) + '\n'
                else:
                    content = str(values[f]) + ','
                fileObject.write(content)


def ReadQDict(path):
    Q = dict()
    with open(path+'/q.txt') as fileObject:
        for line in fileObject.readlines():
            arr = line.split(',')
            Q[int(arr[0])] = list()
            for i in range(1, len(arr)):
                Q[int(arr[0])].append(float(arr[i]))
    return Q

def WritePDict(path, p, F):
    with open(path+'/p.txt','w') as fileObject:
        for key, values in p.items():
            fileObject.write(str(key) + ',')
            for f in range(F):
                if f == F-1:
                    content = str(values[f]) + '\n'
                else:
                    content = str(values[f]) + ','
                fileObject.write(content)

def ReadPDict(path):
    P = dict()
    with open(path+'/p.txt') as fileObject:
        for line in fileObject.readlines():
            arr = line.split(',')
            P[int(arr[0])] = list()
            for i in range(1,len(arr)):
                P[int(arr[0])].append(float(arr[i]))
    return P


def WriteBuDict(path, bu):
    with open(path+'/bu.txt','w') as fileObject:
        for key, value in bu.items():
            fileObject.write(str(key)+',')
            fileObject.write(str(value)+'\n')

def ReadBudict(path):
    bu = dict()
    with open(path+'/bu.txt') as fileObject:
        for line in fileObject.readlines():
            arr = line.split(',')
            bu[int(arr[0])] = float(arr[1])
    return bu


def WriteBiDict(path, bi):
    with open(path+'/bi.txt','w') as fileObject:
        for key, value in bi.items():
            fileObject.write(str(key)+',')
            fileObject.write(str(value)+'\n')

def ReadBiDict(path):
    bi = dict()
    with open(path+'/bu.txt') as fileObject:
        for line in fileObject.readlines():
            arr = line.split(',')
            bi[int(arr[0])] = float(arr[1])
    return bi
