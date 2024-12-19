import numpy as np
import pandas as pd
import time
import ACO
from plot import plotPath
from distance import calculateDistance
from all3 import adaptedLargeLocalSearch
from path import rearrange_path
# from ga import ga
# np.seterr(invalid='ignore')



if __name__ == "__main__":
    # 导入数据
    dist_matrix1_path="distance_matrix.csv"
    dist_matrix1 = np.genfromtxt(dist_matrix1_path, delimiter=',', dtype=int)
    city_number = len(dist_matrix1)

    # 初始化距离矩阵dis_mtx、弗洛蒙矩阵tau_mtx、能见度矩阵eta_mtx、选择概率矩阵prob_mtx、分子矩阵numerator_mtx
    dis_mtx = np.zeros((city_number, city_number))
    tau_mtx = np.ones((city_number, city_number))
    eta_mtx = np.zeros((city_number, city_number))
    prob_mtx = np.zeros((city_number, city_number))
    numerator_mtx = np.zeros((city_number, city_number))
    
    # tau_mtx+=ga(prob, mutationProb, population, gaIter, MaxGaIter)
    minV = float('inf')  # 记录矩阵中的最小值，用于噪音公式
    for i in range(len(dis_mtx)):
        for j in range(len(dis_mtx)):
            dis_mtx[i, j] = ACO.distance_calculation(i, j, dist_matrix1)
            if dis_mtx[i, j] !=0 and dis_mtx[i, j] < minV:
                minV = dis_mtx[i, j]

    for i in range(len(eta_mtx)):
        for j in range(len(eta_mtx)):
            if dis_mtx[i, j] != 0:
                eta_mtx[i, j] = 1 / dis_mtx[i, j]

    # 初始化模型相关参数
    rounds = 100 # 轮数
    ant_number = 225  # 蚂蚁数
    alpha = 2  # =0时容易陷入局部最优解，信息素指数
    beta = 5  # =0时收敛过快，也无法最优，可见度指数
    rho = 0.15  # 消散系数
    Q = 60  # 佛罗蒙更新常量
    K = 300  # 大邻域算法的次数
    destroyDegree = 0.2  # 破坏度
    lowwer=1#决策变量下界
    upper=1#决策变量上界
    dim=151#决策变量维度
    mut_way=0.6#变异几率
    epochs=5#迭代次数
    CR =0.3#交叉概率
    k=7
    prob, mutationProb, population, gaIter, MaxGaIter = 0.8, 0.08, 40, 0, 100
    save = 10
    # 调用函数进行求解
    tic = time.perf_counter()
    # optimal_policy, shortest_dis_lst = ACO3.acoRound(rounds, tau_mtx, rho, ant_number, city_number, eta_mtx, alpha, beta, dis_mtx, Q,prob, mutationProb, population, gaIter, MaxGaIter,save)  # 遗传算法
    optimal_policy, shortest_dis_lst = ACO.acoRound(rounds,tau_mtx,rho,ant_number,city_number, eta_mtx,alpha,beta,dis_mtx,Q,population,lowwer,upper,dim,mut_way,epochs,save,k,CR)  # 进化算法
    optimal_dis = min(shortest_dis_lst)
    print(optimal_dis)
    

    print(f"最短路径为{calculateDistance(optimal_policy,dis_mtx,circle=True)/10000}千米")
    optimal_path, optimal_dis1, scoreLst = adaptedLargeLocalSearch(optimal_policy, K, destroyDegree)  # 大邻域
    optimal_path = optimal_path if optimal_dis1 < optimal_dis else optimal_policy
    print(optimal_path)
    optimal_path = rearrange_path(optimal_path, 115, 116)  # 从0开始
    toc = time.perf_counter()
    print(f"最短路径为{calculateDistance(optimal_path,dis_mtx,circle=False)/10000}千米")
    print(f"最优策略为{optimal_path}")
    print(f"计算耗时:{toc - tic:0.4f}秒")
    print("各算子得分情况为:",scoreLst)
    print(len(optimal_path))
    # 使用txt存储结果
    file_save = open(r"C:\Users\HONOR\Music\fin\jieguo.txt", 'w').close()
    file_save_handle = open(r'C:\Users\HONOR\Music\fin\jieguo.txt', mode='a')
    for i in range(0, len(optimal_path)):
        file_save_handle.write(str(optimal_path[i]))
        file_save_handle.write('\n')
    file_save_handle.close()
    location_path = "coordinates.json"
    import json
    with open(location_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
    location = np.array(list(data.values()))
    data_df = pd.DataFrame(location, columns=['longitude','latitude']) 
    plotPath(optimal_path,data_df)