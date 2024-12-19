import copy
import random
import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
import sys
import time
from numba import njit

from distance import calculateDistance
from GA import _run
# from de2 import Gena_TSP
from kmeans import kmeans
from all3 import adaptedLargeLocalSearch
# np.seterr(invalid='ignore')
a = 0
from multiprocessing import Pool
from randchoice import random_selection
# 显示求解进度
# @njit(cache=True)
def progress_bar(process_rate):
    process = int(100*process_rate//1+1)
    print("\r", end="")
    print("计算进度: {}%: ".format(process), "▋" * (process // 2), end="")
    sys.stdout.flush()


# 根据经纬度输出实际球面距离:输入两点在df中的索引，输出两点的球面距离
@njit(cache=True)
def distance_calculation(i_index, j_index, dist_matrix1):

    return dist_matrix1[i_index][j_index]


# 按照轮盘赌算法随机选择下一个目标:输入各个概率组成的列表，输出所选择对象的索引



# 计算每一个OD结点对的选择概率:输入弗洛蒙浓度矩阵tau_m(tau_mtx)、能见度矩阵eta_mtx(eta_mtx)、搜索禁忌列表taboo_l(taboo_lst)、当前结点索引，输出选择概率列表

def calculate_probability(tau_m, eta_m, taboo_l, current_index,city_number,alpha,beta):
    numerator_lst = np.zeros(city_number)
    prob_lst = np.zeros(city_number)
    for i in range(0, len(taboo_l)):
        if taboo_l[i] == 1:  # 根据禁忌表中，可以访问的城市，访问过的城市为0，没有的为1，1表示可以访问
            numerator_lst[i] = (tau_m[current_index, i] ** alpha) * (eta_m[current_index, i] ** beta)+0.1**50  # 分子的计算
    prob_lst = numerator_lst / np.sum(numerator_lst)
        
    return prob_lst

def ant_colony_optimization_parallel(args):
    start_index, tau_m, city_number, eta_mtx, alpha, beta = args
    return ant_colony_optimization(start_index, tau_m, city_number, eta_mtx, alpha, beta)
# 蚁群算法，单只蚂蚁的视角:输入起点，和初始弗洛蒙矩阵，不重复地走完所有其他结点，最后回到起点，输出此时的距离dis

def ant_colony_optimization(start_index, tau_m,city_number, eta_mtx,alpha,beta):
    current_index = start_index
    path_index = []  # 用于存储走过的【结点】索引
    taboo_lst = [1 for i in range(0, city_number)]
    taboo_lst[start_index] = 0
    path_index.append(current_index)
    while sum(taboo_lst) != 0:  # 不重复地访问除了起点之外的所有结点
        next_index = random_selection(calculate_probability(tau_m, eta_mtx, taboo_lst, current_index,city_number,alpha,beta))  # 下一个结点的索引
        taboo_lst[next_index] = 0
        path_index.append(next_index)
        current_index = next_index
    # path_index.append(start_index)  # 由若干结点的索引，构成的路径
    return path_index

def acoRound(rounds,tau_mtx,rho,ant_number,city_number, eta_mtx,alpha,beta,dis_mtx,Q,population,lowwer,upper,dim,mut_way,epochs,save,k,CR):
# 蚁群算法，多轮次
# def acoRound(rounds,tau_mtx,rho,ant_number,city_number, eta_mtx,alpha,beta,dis_mtx,Q,prob, mutationProb, population, gaIter, MaxGaIter):

    # all_elements = set(range(k))
    # labels,reachable = kmeans(k)
    # unreachable = [all_elements - s for s in reachable]
    # for class_label in range(len(unreachable)):
    #     unreachable_labels = unreachable[class_label]
    #     # 获取当前标签对应的所有节点索引
    #     indices = np.where(labels == class_label)[0]
    #     for unreachable_label in unreachable_labels:
    #         # 获取不可达标签对应的所有节点索引
    #         unreachable_indices = np.where(labels == unreachable_label)[0]
    #         # 将 tau_mtx 中对应位置的值设置为 0
    #         tau_mtx[np.ix_(indices, unreachable_indices)] = 0
           
    average_dis_lst = set()  # 单轮次所有蚂蚁的平均距离
    shortest_dis_lst = set()   # 单轮次所有蚂蚁的最短距离
    shortest_till_now_lst =set()  # 用来储存截止到目前的最短距离
    optimal_policy_round = []  # 初始化单轮次内，蚂蚁的最优行驶路径
    optimal_policy = []  # 初始化全局次内，蚂蚁的最优行驶路径
    times = 1  # 仿真轮次初始值
    pool = Pool(processes=16)
    tic = time.perf_counter()
    while times <= rounds:
        
        if times >= 1:
            ant_number = ant_number
        #     tau_mtx+=ga(prob, mutationProb, population, gaIter, MaxGaIter,save)
        if times == 1:
            
            taup,optimalp,shortest,top_individuals=_run(calculateDistance, constraints, lowwer, 0, population, dim, mut_way, epochs, save, Q, k, CR, pop=None)
            tau_mtx+=taup
            if shortest_dis_lst is not None or shortest<min(shortest_dis_lst):
                shortest_dis_lst.add(shortest)
                shortest_till_now = shortest  # 截止到目前的最短距离
                shortest_till_now_lst.add(shortest_till_now)  # 用来储存截止到目前的最短距离
                optimal_policy=optimalp
                shortest_till_now_lst.add(shortest)
                shortest_dis_lst.add(shortest)
                print(shortest_till_now,'out')
            
        policy_mtx = []  # 初始化每只蚂蚁的行驶路径，里面具有蚂蚁数量个的路径（每一个都是由结点序列组成的列表）
        sigle_round_dis_lst = []  # 单轮次中，记录每只蚂蚁的访问总距离
        progress_bar(times / rounds)
        # tau_mtx_round = copy.deepcopy(tau_mtx)  # 在同一轮的概率计算中所使用的不变的弗洛蒙矩阵
        # tau_mtx = copy.deepcopy(tau_mtx * rho)
        
        tau_mtx_round = copy.deepcopy(tau_mtx)  
        tau_mtx = copy.deepcopy(tau_mtx * rho)  
        # 并行计算每一只蚂蚁的路径
        args = [(random.randint(0, city_number-1), tau_mtx_round, city_number, eta_mtx, alpha, beta) for _ in range(ant_number)]
        policy_mtx = pool.map(ant_colony_optimization_parallel, args)

        for m in range(0, len(policy_mtx)):  # 每一只蚂蚁的TSP
            path = policy_mtx[m]  # 导出第k个蚂蚁路径[0,17,....,0],len(path)=102
            distance = calculateDistance(path, dis_mtx)  # 计算路径的总距离
            sigle_round_dis_lst.append(distance)  # 存储了每一只蚂蚁的行驶距离

            delta_tau = Q / distance  # 增加的弗洛蒙值
            for i in range(0, len(path)-1):
                tau_mtx[path[i], path[i+1]] += delta_tau  # 更新弗洛蒙矩阵
            tau_mtx[path[-1], path[0]] += delta_tau  # 更新弗洛蒙矩阵
   
        average_dis_lst.add(np.average(sigle_round_dis_lst))  # 用来存储这一轮中n个蚂蚁的平均行驶距离
        shortest_dis_lst.add(min(sigle_round_dis_lst))  # 用来存储这一轮中n个蚂蚁的最短距离
        shortest_till_now = min(shortest_dis_lst)  # 截止到目前的最短距离
        shortest_till_now_lst.add(shortest_till_now)  # 用来储存截止到目前的最短距离
        optimal_policy_index = sigle_round_dis_lst.index(min(sigle_round_dis_lst))  # 找到当前轮次最短行驶距离对应的蚂蚁索引
        optimal_policy_round = policy_mtx[optimal_policy_index]
        print(sigle_round_dis_lst,'yiqun')
        if min(sigle_round_dis_lst) == shortest_till_now:
            
            optimal_policy = optimal_policy_round  # 此时该轮最优策略即为optimal_policy
            # optimal_policy = local_search(optimal_policy)
            print(shortest_till_now,'outshang')
            optimal_pol1,optimal_dis, scoreLst= adaptedLargeLocalSearch(optimal_policy, 100,0.2)
            optimal_policy = optimal_pol1 if optimal_dis < min(shortest_dis_lst) else optimal_policy
            if  optimal_dis < min(shortest_dis_lst) :
                optimal_policy = optimal_pol1
            shortest_dis_lst.add(optimal_dis)
            shortest_till_now = optimal_dis  # 截止到目前的最短距离
            shortest_till_now_lst.add(shortest_till_now)  # 用来储存截止到目前的最短距离
            shortest_dis_lst.add(optimal_dis)
        siglesort = np.argsort(sigle_round_dis_lst)[:save]
        
        # if times != 1:
            
        times += 1
        # tau_mtx+=ga(prob, mutationProb, population, gaIter, MaxGaIter,save,[policy_mtx[i] for i in siglesort])
        # tau_mtx+=
        
    

    # 替换集合中的元素
        if shortest_till_now<3000000 and times%1==0:
            CR = 0.8
            mut_way = 1
            taup,optimalp,shortest,top_individuals=_run(calculateDistance, constraints, lowwer, 0, population, dim, mut_way, epochs, save, Q, k, CR, pop=top_individuals)
            tau_mtx+=taup
            if shortest<min(shortest_dis_lst):
                shortest_dis_lst.add(shortest)
                shortest_till_now = shortest  # 截止到目前的最短距离
                shortest_till_now_lst.add(shortest_till_now)  # 用来储存截止到目前的最短距离
                optimal_policy=optimalp
                shortest_till_now_lst.add(shortest)
                shortest_dis_lst.add(shortest)
                print(shortest_till_now,'out')
        toc = time.perf_counter()
        print(f"zhuyao1计算耗时:{toc - tic:0.4f}秒")
    
    pool.close()
    pool.join()
    print(optimal_policy,"over")
    return np.array(optimal_policy),shortest_dis_lst




# 绘制路线,输入最优策略和节点文件，输出路线图示
def plotPath(optimal_policy,node_file):
    pointsX = []
    pointsY = []
    for i in range(0,len(optimal_policy)-1):
        Node = optimal_policy[i]
        NodeX = node_file.loc[Node, 'longitude']
        NodeY = node_file.loc[Node, 'latitude']
        pointsX.append(NodeX)
        pointsY.append(NodeY)
    pointsX.append(node_file.loc[optimal_policy[0], 'longitude'])
    pointsY.append(node_file.loc[optimal_policy[0], 'latitude'])
    fig, ax = plt.subplots(figsize=(20, 15), dpi=100)
    ax = plt.scatter(pointsX, pointsY, color = 'red')
    ax = plt.plot(pointsX, pointsY, color = 'blue')
    for i in range(0, len(node_file)):
        locX = node_file.loc[i, 'longitude'] + 0.25
        locY = node_file.loc[i, 'latitude'] + 0.25
        label = i
        plt.text(locX, locY, str(label), family='serif', style='italic', fontsize=15, verticalalignment="bottom", ha='left', color='k')
    plt.savefig(r"C:\Users\honor\music\litian.png")
    plt.show()


# 将tsp的起终点调整从0到0
# @njit(cache=True)


def constraints(x):
    return 0





