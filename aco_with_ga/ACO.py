import copy
import random
import sys
import time


from distance import calculateDistance
from GA import _run
from all3 import adaptedLargeLocalSearch
from multiprocessing import Pool
from ACOfunction import ant_colony_optimization
# 显示求解进度
# @njit(cache=True)
def progress_bar(process_rate):
    process = int(100*process_rate//1)
    print("\r", end="")
    print("计算进度: {}%: ".format(process), "▋" * (process // 2), end="")
    sys.stdout.flush()


# 根据经纬度输出实际球面距离:输入两点在df中的索引，输出两点的球面距离



# 按照轮盘赌算法随机选择下一个目标:输入各个概率组成的列表，输出所选择对象的索引



# 计算每一个OD结点对的选择概率:输入弗洛蒙浓度矩阵tau_m(tau_mtx)、能见度矩阵eta_mtx(eta_mtx)、搜索禁忌列表taboo_l(taboo_lst)、当前结点索引，输出选择概率列表



def ant_colony_optimization_parallel(args):
    start_index, tau_m, city_number, eta_mtx, alpha, beta = args
    return ant_colony_optimization(start_index, tau_m, city_number, eta_mtx, alpha, beta)
# 蚁群算法，单只蚂蚁的视角:输入起点，和初始弗洛蒙矩阵，不重复地走完所有其他结点，最后回到起点，输出此时的距离dis



def acoRound(rounds, tau_mtx, rho, ant_number, city_number, eta_mtx, alpha, beta, dis_mtx, Q, population, lowwer, upper, dim, mut_way, epochs, save, k, CR):
    # 蚁群算法，多轮次
    import numpy as np
    average_dis_lst = np.array([])  # 单轮次所有蚂蚁的平均距离
    shortest_dis_lst = np.array([])   # 单轮次所有蚂蚁的最短距离
    shortest_till_now_lst = np.array([])  # 用来储存截止到目前的最短距离
    optimal_policy_round = np.array([])  # 初始化单轮次内，蚂蚁的最优行驶路径
    optimal_policy = np.array([])  # 初始化全局次内，蚂蚁的最优行驶路径
    times = 1  # 仿真轮次初始值
    pool = Pool(processes=16)
    tic = time.perf_counter()
    while times <= rounds:
        if times >= 1:
            ant_number = ant_number
        if times == 1:
            taup, optimalp, shortest, top_individuals = _run(calculateDistance, constraints, lowwer, 0, population, dim, mut_way, epochs, save, Q, k, CR, pop=None)
            tau_mtx += taup
            if shortest_dis_lst.size == 0 or shortest < shortest_dis_lst.min():
                shortest_dis_lst = np.append(shortest_dis_lst, shortest)
                shortest_till_now = shortest  # 截止到目前的最短距离
                shortest_till_now_lst = np.append(shortest_till_now_lst, shortest_till_now)  # 用来储存截止到目前的最短距离
                optimal_policy = optimalp
                shortest_till_now_lst = np.append(shortest_till_now_lst, shortest)
                shortest_dis_lst = np.append(shortest_dis_lst, shortest)
                # print(shortest_till_now, 'out')
        policy_mtx = np.array([])  # 初始化每只蚂蚁的行驶路径
        sigle_round_dis_lst = np.array([])  # 单轮次中，记录每只蚂蚁的访问总距离
        progress_bar(times / rounds)
        tau_mtx_round = copy.deepcopy(tau_mtx)
        tau_mtx = copy.deepcopy(tau_mtx * rho)
        args = [(random.randint(0, city_number - 1), tau_mtx_round, city_number, eta_mtx, alpha, beta) for _ in range(ant_number)]
        policies = pool.map(ant_colony_optimization_parallel, args)
        policy_mtx = np.array(policies)
        for m in range(0, len(policy_mtx)):  # 每一只蚂蚁的TSP
            path = policy_mtx[m]  # 导出第k个蚂蚁路径
            distance = calculateDistance(path, dis_mtx)  # 计算路径的总距离
            sigle_round_dis_lst = np.append(sigle_round_dis_lst, distance)
            delta_tau = Q / distance  # 增加的弗洛蒙值
            for i in range(0, len(path) - 1):
                tau_mtx[path[i], path[i + 1]] += delta_tau  # 更新弗洛蒙矩阵
            tau_mtx[path[-1], path[0]] += delta_tau  # 更新弗洛蒙矩阵
        average_dis_lst = np.append(average_dis_lst, np.average(sigle_round_dis_lst))  # 存储这一轮中蚂蚁的平均行驶距离
        shortest_dis_lst = np.append(shortest_dis_lst, sigle_round_dis_lst.min())  # 存储这一轮中蚂蚁的最短距离
        shortest_till_now = shortest_dis_lst.min()  # 截止到目前的最短距离
        shortest_till_now_lst = np.append(shortest_till_now_lst, shortest_till_now)  # 用来储存截止到目前的最短距离
        optimal_policy_index = np.argmin(sigle_round_dis_lst)  # 找到当前轮次最短行驶距离对应的蚂蚁索引
        optimal_policy_round = policy_mtx[optimal_policy_index]
        # print(np.min(sigle_round_dis_lst), 'yiqun')
        if sigle_round_dis_lst.min() == shortest_till_now:
            optimal_policy = np.array(optimal_policy_round).astype(np.int64)  # 此时该轮最优策略即为optimal_policy
            # print(shortest_till_now, 'outshang')
            # print(type(optimal_policy))
            optimal_pol1, optimal_dis, scoreLst = adaptedLargeLocalSearch(optimal_policy, 100, 0.2, scorelen=20)
            if optimal_dis < shortest_dis_lst.min():
                optimal_policy = optimal_pol1
            shortest_dis_lst = np.append(shortest_dis_lst, optimal_dis)
            shortest_till_now = optimal_dis  # 截止到目前的最短距离
            shortest_till_now_lst = np.append(shortest_till_now_lst, shortest_till_now)
            shortest_dis_lst = np.append(shortest_dis_lst, optimal_dis)
        siglesort = np.argsort(sigle_round_dis_lst)[:save]
        top10_indices = siglesort[:save]  # 获取前 5 个索引
        top_individuals = np.concatenate((top_individuals,policy_mtx[top10_indices]), axis=0)
        times += 1
        if shortest_till_now < 3000000 and times % 1 == 0:
            CR = 0.8
            mut_way = 1
            taup, optimalp, shortest, top_individuals = _run(calculateDistance, constraints, lowwer, 0, population, dim, mut_way, epochs, save, Q, k, CR, pop=top_individuals)
            tau_mtx += taup
            if shortest < shortest_dis_lst.min():
                shortest_dis_lst = np.append(shortest_dis_lst, shortest)
                shortest_till_now = shortest  # 截止到目前的最短距离
                shortest_till_now_lst = np.append(shortest_till_now_lst, shortest_till_now)
                optimal_policy = optimalp
                shortest_till_now_lst = np.append(shortest_till_now_lst, shortest)
                shortest_dis_lst = np.append(shortest_dis_lst, shortest)
                # print(shortest_till_now, 'out')
        toc = time.perf_counter()
        print(f"zhuyao1计算耗时:{toc - tic:0.4f}秒")
    pool.close()
    pool.join()
    # print(optimal_policy, "over")
    return np.array(optimal_policy), shortest_dis_lst




# 绘制路线,输入最优策略和节点文件，输出路线图示


# 将tsp的起终点调整从0到0
# @njit(cache=True)


def constraints(x):
    return 0





