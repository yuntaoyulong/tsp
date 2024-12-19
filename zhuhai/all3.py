
import numpy as np
import math
import random
from numba import njit
from delete import delete_workaround
from distance import calculateDistance
dist_matrix1_path="distance_matrix.csv"
dis_mtx = np.genfromtxt(dist_matrix1_path, delimiter=',', dtype=int)
minV = float('inf') 



# 定义adaptedLargeLocalSearch，使用的随机删除，准则为SA的Metropolis接受准则
@njit(cache=True)
def adaptedLargeLocalSearch(feasibleSolve, maxInteration, destroyDegree,scorelen =24):
    lamb = 0.7
    nodeLst = feasibleSolve
    alpha = 0.8  # 降温系数
    currentT = 300  # 当前温度
    scoreLst = np.ones(scorelen) # 权重列表（得分列表）,破坏和修复算子+2邻边搜索，一共13种组合，0-（0，0）、1-（0，1）、以此类推，最后一个是2邻边

    for k in range(0, maxInteration):  # 每个温度下需要进行一次破坏+修复，共生成maxInteration个邻域
        ei = calculateDistance(nodeLst, dis_mtx)  # 先前解的值
        # neighborMethodIndex = random_selection(scoreLst)
        neighborMethodIndex = np.random.randint(0, scorelen)
        if neighborMethodIndex == 0:
            nodeLst, delateNodeLst = destroyOperator_randomRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_greedyInsertion(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex == 1:
            nodeLst, delateNodeLst = destroyOperator_randomRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_randomInsertion(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex == 2:
            nodeLst, delateNodeLst = destroyOperator_randomRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_greedyInsertionWithNoise(nodeLst, delateNodeLst)  # 修复

        elif neighborMethodIndex == 3:
            nodeLst, delateNodeLst = destroyOperator_relatedRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_greedyInsertion(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex == 4:
            nodeLst, delateNodeLst = destroyOperator_relatedRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_randomInsertion(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex == 5:
            nodeLst, delateNodeLst = destroyOperator_relatedRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_greedyInsertionWithNoise(nodeLst, delateNodeLst)  # 修复

        elif neighborMethodIndex == 6:
            nodeLst, delateNodeLst = destroyOperator_clusterRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_greedyInsertion(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex == 7:
            nodeLst, delateNodeLst = destroyOperator_clusterRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_randomInsertion(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex == 8:
            nodeLst, delateNodeLst = destroyOperator_clusterRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_greedyInsertionWithNoise(nodeLst, delateNodeLst)  # 修复

        elif neighborMethodIndex == 9:
            nodeLst, delateNodeLst = destroyOperator_worstRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_greedyInsertion(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex == 10:
            nodeLst, delateNodeLst = destroyOperator_worstRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_randomInsertion(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex == 11:
            nodeLst, delateNodeLst = destroyOperator_worstRemoval(feasibleSolve, destroyDegree)  # 破坏
            nodeLst = repairOperator_greedyInsertionWithNoise(nodeLst, delateNodeLst)  # 修复
        elif neighborMethodIndex >= 12:
            nodeLst = twoOpt(feasibleSolve,maxInteration)  # 2-邻边搜索

        ej = calculateDistance(nodeLst,dis_mtx)  # 修复后解的值
        acceptFlag = metropolisCirterion(ei, ej, currentT)
        if acceptFlag == 1:  # 接受
            feasibleSolve = nodeLst
        else:
            pass
        currentT = alpha * currentT

        # 根据解的情况更新分数
        if ej < ei:
            scoreLst[neighborMethodIndex] = lamb*scoreLst[neighborMethodIndex] + 3*(1-lamb)
        elif acceptFlag == 1:
            scoreLst[neighborMethodIndex] = lamb*scoreLst[neighborMethodIndex] + 2*(1-lamb)
        elif acceptFlag == 0:
            scoreLst[neighborMethodIndex] = lamb*scoreLst[neighborMethodIndex] + 1*(1-lamb)
    distance = calculateDistance(feasibleSolve,dis_mtx)
    return feasibleSolve, distance,scoreLst
@njit(cache=True)
def update_score(score, lamb, ej, ei, acceptFlag):
    if ej < ei:
        return lamb * score + 3 * (1 - lamb)
    elif acceptFlag == 1:
        return lamb * score + 2 * (1 - lamb)
    else:
        return lamb * score + 1 * (1 - lamb)

# 定义相关性，这里认为欧式距离最远，两个点的相关性最小，输入节点的
@njit(cache=True)
def calculateRelevance(i,j):
    relevance = -math.exp(dis_mtx[i][j]/100000)
    return relevance


"""________________________________________________破坏算子【4个】______________________________________________________"""
# 破坏算子:【随机删除】，输入可行解、破坏度，输出被破坏后的解、所破坏的节点集合

@njit(cache=True)
def destroyOperator_randomRemoval(feasibleSolve, destroyDegree):
    destrotNum = np.int64(len(feasibleSolve) * destroyDegree)
    nodeLst = np.copy(feasibleSolve)
    return _destroyOperator_randomRemoval(nodeLst, destrotNum)

@njit(cache=True)
def _destroyOperator_randomRemoval(nodeLst, destrotNum):
    delateNodeLst = np.empty(0, dtype=np.int64)
    for i in range(destrotNum):
        randomIndex = np.random.randint(len(nodeLst))
        randomNode = np.int64(nodeLst[randomIndex])
        nodeLst=delete_workaround(nodeLst,randomNode)  # 使用 NumPy 的删除操作
        delateNodeLst=np.append(delateNodeLst,randomNode)
    return nodeLst, delateNodeLst


# 破坏算子:【相关删除：依次删除相关性小的节点】，输入可行解、破坏度，输出被破坏后的解、所破坏的节点集合

@njit(cache=True)
def destroyOperator_relatedRemoval(feasibleSolve,destroyDegree):
    destrotNum = np.int64(len(feasibleSolve) * destroyDegree)
    delateNodeLst = np.empty(0, dtype=np.int64)
    nodeLst = np.copy(feasibleSolve)
    # 破坏算子
    randomNodeSeednum = np.int64(np.random.randint(0,len(nodeLst)))
    randomNodeSeed = np.int64(nodeLst[randomNodeSeednum])
    delateNodeLst= np.append(delateNodeLst,randomNodeSeed)
    nodeLst=delete_workaround(nodeLst,randomNodeSeed)
    currentNode = randomNodeSeed  # 初始化
    aimNode = currentNode
    while len(delateNodeLst) < destrotNum:
        relevance = 0
        for j in nodeLst: # 遍历寻找与currentNode相关性最小的节点aimNode
            while calculateRelevance(currentNode,j) < relevance:
                relevance = calculateRelevance(currentNode,j)
                aimNode = np.int64(j)
                break
        delateNodeLst= np.append(delateNodeLst,aimNode)
        nodeLst=delete_workaround(nodeLst,aimNode)
        currentNode = aimNode
    return nodeLst,delateNodeLst


# 破坏算子:【聚类删除：按照地理距离聚类，最近的2个里面随机选1个删除】，输入可行解、破坏度，输出被破坏后的解、所破坏的节点集合

@njit(cache=True)
def destroyOperator_clusterRemoval(feasibleSolve, destroyDegree):
    destrotNum = np.int64(len(feasibleSolve) * destroyDegree)
    nodeLst = np.copy(feasibleSolve)
    return _destroyOperator_clusterRemoval(nodeLst, destrotNum, dis_mtx)

@njit(cache=True)
def _destroyOperator_clusterRemoval(nodeLst, destrotNum, dis_mtx):
    delateNodeLst = np.empty(0, dtype=np.int64)
 # 随机选择一个初始种子
    randomNodeSeednum = np.int64(np.random.randint(0,len(nodeLst)))
    randomNodeSeed = nodeLst[randomNodeSeednum]
    delateNodeLst= np.append(delateNodeLst,randomNodeSeed)
    nodeLst=delete_workaround(nodeLst,randomNodeSeed)
    currentNode = randomNodeSeed  # 初始化
    while len(delateNodeLst) < destrotNum:
        minValue = float('inf')
        Nodelst = []  # 用来记录距离下降过程中的节点
        for j in nodeLst:  # 遍历寻找与currentNode距离最近的两个节点aimNodes
            if dis_mtx[currentNode][j] < minValue:
                minValue = dis_mtx[currentNode][j]
                aimNode = j
                Nodelst.append(aimNode)
        minTopTwo = Nodelst[-2:]
        randomNodeSeednum1 = np.int64(np.random.randint(0,len(minTopTwo)))
        aimNode = np.int64(minTopTwo[randomNodeSeednum1])
        delateNodeLst= np.append(delateNodeLst,aimNode)
        nodeLst=delete_workaround(nodeLst,aimNode)
        currentNode = aimNode
    return nodeLst, delateNodeLst


# 破坏算子:【最坏删除：依次删除成本最大的节点】，输入可行解、破坏度，输出被破坏后的解、所破坏的节点集合，相当于最大增益
@njit(cache=True)
def destroyOperator_worstRemoval(feasibleSolve, destroyDegree):
    destrotNum = int(len(feasibleSolve) * destroyDegree)
    nodeLst = np.copy(feasibleSolve)
    # nodeLst = np.array(nodeLst[0:-1])
    return _destroyOperator_worstRemoval(nodeLst, destrotNum)

@njit(cache=True)
def _destroyOperator_worstRemoval(nodeLst, destrotNum):
    delateNodeLst = np.empty(0, dtype=np.int64)
    while len(delateNodeLst) < destrotNum:
        maxSavings = -float('inf')
        aimNode = -1
        for j in range(1, len(nodeLst) - 1):
            costPre = dis_mtx[nodeLst[j - 1], nodeLst[j]] + dis_mtx[nodeLst[j], nodeLst[j + 1]]
            costAfter = dis_mtx[nodeLst[j - 1], nodeLst[j + 1]]
            if costPre - costAfter > maxSavings:
                maxSavings = costPre - costAfter
                aimNode = np.int64(nodeLst[j])
        delateNodeLst= np.append(delateNodeLst,aimNode)
        nodeLst=delete_workaround(nodeLst,aimNode)
    return nodeLst, delateNodeLst

"""_________________________________________________修复算子【3个】_____________________________________________________"""
# 修复算子:【贪婪插入】，输入被破坏后的解、所破坏的节点集合，以贪婪插入的方式修复解，将所被破坏的解插回被破坏后的解中，输出新可行解

@njit(cache=True)
def repairOperator_greedyInsertion(nodeLst, delateNodeLst):
    while len(delateNodeLst):
        currentNode = delateNodeLst[-1]
        delateNodeLst = delateNodeLst[:-1]
        temp = []
        temp_index = []
        flag = 0  # 待插入节点i是否已经插入进去了，1是，0没有
        for j in range(0, len(nodeLst)):
            if j == 0:
                lengthAC = dis_mtx[0][currentNode]
                lengthCB = dis_mtx[currentNode][nodeLst[j]]
                lengthAB = dis_mtx[0][nodeLst[j]]
                if lengthAC + lengthCB <= lengthAB:  # 距离小于则插入
                    nodeLst=np.concatenate((nodeLst[:j], np.array([currentNode]), nodeLst[j:]))
                    flag = 1
                    break
                else:  # 距离大于则记录，然后最终再插入最小的地方
                    temp.append(lengthAC + lengthCB)
                    temp_index.append(j)
            if j == len(nodeLst):
                lengthAC = dis_mtx[nodeLst[j]][currentNode]
                lengthCB = dis_mtx[currentNode][0]
                lengthAB = dis_mtx[nodeLst[j]][0]
                if lengthAC + lengthCB <= lengthAB:  # 距离小于则插入
                    nodeLst=np.concatenate((nodeLst[:j], np.array([currentNode]), nodeLst[j:]))
                    flag = 1
                    break
                else:  # 距离大于则记录，然后最终再插入最小的地方
                    temp.append(lengthAC + lengthCB)
                    temp_index.append(j)
            if j != 0 and j != len(nodeLst):
                lengthAC = dis_mtx[nodeLst[j - 1]][currentNode]
                lengthCB = dis_mtx[currentNode][nodeLst[j]]
                lengthAB = dis_mtx[nodeLst[j - 1]][nodeLst[j]]
                if lengthAC + lengthCB <= lengthAB:  # 距离小于则插入
                    nodeLst=np.concatenate((nodeLst[:j], np.array([currentNode]), nodeLst[j:]))
                    
                    flag = 1
                    break
                else:  # 距离大于则记录，然后最终再插入最小的地方
                    temp.append(lengthAC + lengthCB)
                    temp_index.append(j)
        if flag == 0:
            optimal_index = temp.index(min(temp))
            nodeLst=np.concatenate((nodeLst[:optimal_index], np.array([currentNode]), nodeLst[optimal_index:]))
            
        else:
            pass
   # nodeLst.insert(0, 0)
    # nodeLst=np.append(nodeLst,nodeLst[0])
    return nodeLst


# 修复算子:【随机插入】，输入被破坏后的解、所破坏的节点集合，以随机插入的方式修复解，将所被破坏的解插回被破坏后的解中，输出新可行解

@njit(cache=True)
def repairOperator_randomInsertion(nodeLst, delateNodeLst):

    while len(delateNodeLst):
       # print(insertIndexLst,nodeLst)
        currentNode = delateNodeLst[-1]
        delateNodeLst = delateNodeLst[:-1]
        insertIndex = np.random.randint(0, len(nodeLst))
        nodeLst =  np.concatenate((nodeLst[:insertIndex], np.array([currentNode]), nodeLst[insertIndex:]))
    
   # nodeLst.insert(0, 0)
    # nodeLst=np.append(nodeLst,nodeLst[0])
    return nodeLst


# 修复算子:【带噪音的贪婪插入】，输入被破坏后的解、所破坏的节点集合，以带噪音的贪婪插入方式修复解，将所被破坏的解插回被破坏后的解中，输出新可行解

@njit(cache=True)
def repairOperator_greedyInsertionWithNoise(nodeLst, delateNodeLst):
    while len(delateNodeLst) :
        currentNode = delateNodeLst[-1]
        # print(currentNode)
        delateNodeLst = delateNodeLst[:-1]
        temp = []
        temp_index = []
        flag = 0  # 待插入节点i是否已经插入进去了，1是，0没有
        for j in range(len(nodeLst)):
            if j == 0:
                lengthAC = dis_mtx[0][currentNode]
                lengthCB = dis_mtx[currentNode][nodeLst[j]]
                lengthAB = dis_mtx[0][nodeLst[j]]
                noise = 0.1* random.uniform(-1,1)*minV
                if lengthAC + lengthCB + noise <= lengthAB:  # 距离小于则插入
                    nodeLst=np.concatenate((nodeLst[:j], np.array([currentNode]), nodeLst[j:]))
                    flag = 1
                    break
                else:  # 距离大于则记录，然后最终再插入最小的地方
                    temp.append(lengthAC + lengthCB)
                    temp_index.append(j)
            if j == len(nodeLst):
                lengthAC = dis_mtx[nodeLst[j]][currentNode]
                lengthCB = dis_mtx[currentNode][0]
                lengthAB = dis_mtx[nodeLst[j]][0]
                noise = 0.1* random.uniform(-1,1)*minV
                if lengthAC + lengthCB + noise<= lengthAB:  # 距离小于则插入
                    nodeLst=np.concatenate((nodeLst[:j], np.array([currentNode]), nodeLst[j:]))
                    flag = 1
                    break
                else:  # 距离大于则记录，然后最终再插入最小的地方
                    temp.append(lengthAC + lengthCB)
                    temp_index.append(j)
            if j != 0 and j != len(nodeLst):
                lengthAC = dis_mtx[nodeLst[j - 1]][currentNode]
                lengthCB = dis_mtx[currentNode][nodeLst[j]]
                lengthAB = dis_mtx[nodeLst[j - 1]][nodeLst[j]]
                noise = 0.1* random.uniform(-1,1)*minV
                if lengthAC + lengthCB + noise  <= lengthAB:  # 距离小于则插入
                    nodeLst=np.concatenate((nodeLst[:j], np.array([currentNode]), nodeLst[j:]))
                    flag = 1
                    break
                else:  # 距离大于则记录，然后最终再插入最小的地方
                    temp.append(lengthAC + lengthCB)
                    temp_index.append(j)
        if flag == 0:
            optimal_index = temp.index(min(temp))
            nodeLst=np.concatenate((nodeLst[:optimal_index], np.array([currentNode]), nodeLst[optimal_index:]))
        else:
            pass
   # nodeLst.insert(0, 0)
    # nodeLst=np.append(nodeLst,nodeLst[0])
    return nodeLst


"""_____________________________________________________2邻边搜索____________________________________________________"""

@njit(cache=True)
def twoOpt(feasibleSolve, maxInteration):
    # 进行maxInteration次的2-邻边优化算法,调整最终的每一只蚂蚁的TSP
    for k in range(maxInteration):  # 2-邻边算法循环，假设进行maxInteration轮改进，maxInteration个邻域
        flag = 0  # 2-邻边算法的退出标志
        for i in range(len(feasibleSolve) - 4):  # len(path)=102 ; city_number = 101; len(dis_mtx)=101
            for j in range(i + 2, len(feasibleSolve) - 2):
                if (dis_mtx[feasibleSolve[i], feasibleSolve[j]] + dis_mtx[feasibleSolve[i + 1], feasibleSolve[j + 1]]) < (dis_mtx[feasibleSolve[i], feasibleSolve[i + 1]] + dis_mtx[feasibleSolve[j], feasibleSolve[j + 1]]):
                    feasibleSolve[i + 1:j + 1] = feasibleSolve[j:i:-1]  # [i+1:j+1]包括了[i+1,i+2,...,j];[j:i:-1]包括了[j,j-1,...,i+1],切片的左闭右开特性
                    flag = 1
        if flag == 0:
            break
    return feasibleSolve


"""_____________________________________________________接受准则____________________________________________________"""
# Metropolis接受准则，输入原先的值ei，修复后的值ej，温度，输出接受结果

@njit(cache=True)
def metropolisCirterion(ei, ej, temperature):
    if ej < ei:
        return 1  # 无条件接受更优的解
    else:
        prob = math.exp(-(ej - ei) / temperature)  # 计算接受概率
        return 1 if prob > random.random() else 0  # 根据概率决定是否接受新解


# 绘制路线,输入最优策略和节点文件，输出路线图示

# a= np.array([1,2,4,5,6,61,34,54],dtype=np.int64)
# optimal_path, optimal_dis1, scoreLst = adaptedLargeLocalSearch(a, 20, 0.2)  # 大邻域
# print(optimal_path)
