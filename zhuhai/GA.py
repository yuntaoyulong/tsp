import numpy as np
from kmeans import kmeans
from path import generate_path,rearrange_path
import pandas as pd
from distance import calculateDistance
from numba import njit
dist_matrix1_path = "distance_matrix.csv"
dist_matrix1 = np.genfromtxt(dist_matrix1_path, delimiter=',', dtype=int)
cities = len(dist_matrix1)
from plot import plotPath
from GAjit import initpop, constraints, mut, cross, select, get_top_n_individuals
import json
location_path = "coordinates.json"
with open(location_path, 'r', encoding='utf-8') as f1:
    data_json = json.load(f1)
data = np.array(list(data_json.values()))
data1 = data.copy()
import matplotlib.pyplot as plt  
import time  # 用于模拟迭代计算的等待时间  

# 启用matplotlib的交互模式  
if 0:
    plt.ion()  

    fig, ax = plt.subplots()  
    ax.set_xscale('log')  # 设置横坐标为对数尺度  
    x_data, y_data = [], []  


# 假设我们有一个迭代过程，最多迭代1000次  
max_iters = 1000  

data_df = pd.DataFrame(data1, columns=['latitude','longitude']) 
# @njit(cache=True)
def _run(fitness_func, constraints, lowwer, upper, pop_size, dim, mut_way, epochs, save, Q, k, CR, pop=None):
    labels, reachable = kmeans(k-1, data)
    population = np.zeros((pop_size, dim), dtype=np.int64)
    if pop is not None:
        for i in range(pop_size - save):
            population[i] = generate_path(labels, reachable)
        population = np.concatenate((population[:pop_size - len(pop)], pop), axis=0)
    else:
        for i in range(pop_size):
            population[i] = generate_path(labels, reachable)
    for i in range(pop_size):
        path = population[i]
        idx_150 = np.where(path == 150)[0][0]
        idx_116 = np.where(path == 116)[0][0]
        idx_115 = np.where(path == 115)[0][0]
        if abs(idx_150 - idx_116) != 1 or abs(idx_150 - idx_115) != 1:
            path = np.delete(path, [idx_150, idx_116, idx_115])
            insert_pos = np.random.randint(0, len(path) - 2)
            path = np.insert(path, insert_pos, [116, 150, 115] if np.random.rand() > 0.5 else [115, 150, 116])
            
            population[i] = path
    temp = fitness_func(population[pop_size - save], dist_matrix1)
    fit, NFE, best = initpop(population)
    tic = time.perf_counter()
    for i in range(epochs):
        mut_population = mut(population, best, labels, reachable, mut_way, dim)
        
        cross_population = cross(population, mut_population, CR)
        

            
        population, fit = select(cross_population, fit, pop_size,allstime=20,scorelen=24)
        best = population [np.argmin(fit)]
        if 0:
            x_data.append(i)  
            y_data.append(np.min(fit)/10000)  

            # 清除旧的线，并绘制新的线  
            ax.clear()  
            ax.plot(x_data, y_data)  
            
            # 重新设置图表标题和坐标轴标签等  
            ax.set_title("Dynamic Update of Best Value Over Iterations")  
            ax.set_xlabel("Iteration (log scale)")  
            ax.set_ylabel("Best Value(km)")  
            ax.set_xscale('log')  # 确保在清除并重新绘制新线条后，横坐标仍为log scale  
            
            # 强制matplotlib重绘图表  
            plt.draw()  
            plt.pause(0.01)  # 短暂暂停以确保图表得以更新  
        
        # print(population,"haoh")
        top_individuals = get_top_n_individuals(population, fit, pop_size - save)
        
        # if i % 50 == 0:
    toc = time.perf_counter()
    print(np.min(fit))
    print(f"计算耗时:{toc - tic:0.4f}秒")        
    plt.ioff()  

# # 显示最终结果并阻止关闭，直到用户关闭窗口  
    plt.show()  
    # toc = time.perf_counter()
    pheromone = np.zeros((cities, cities)) 
    for i in range(save): 
       
        for j in range(cities - 1):
            if upper:
                pheromone[top_individuals[i][j]][top_individuals[i][j + 1]] += Q / calculateDistance(top_individuals[i],dist_matrix1)
            else:
                pheromone[top_individuals[i][j]][top_individuals[i][j + 1]] += 600000000 / calculateDistance(top_individuals[i],dist_matrix1)

        if upper:
            pheromone[top_individuals[i][-1]][top_individuals[i][0]] += Q / calculateDistance(top_individuals[i],dist_matrix1)
        else:
            pheromone[top_individuals[i][j]][top_individuals[i][j + 1]] += 600000000 / calculateDistance(top_individuals[i],dist_matrix1)
    # print(f"计算耗时:{toc - tic:0.4f}秒")
    # print("best", best, fitness_func(best, dist_matrix1), np.min(fit))
    return pheromone, best, np.min(fit), top_individuals


if __name__ == '__main__':
    

    

    lowwer = 100
    upper = 10
    pop_size = 30
    dim = 151
    mut_way = 0.6
    epochs = 50
    save = 15
    Q = 1
    k = 8
    CR = 0.3
    tic1 = time.perf_counter()
    best1 = np.array([])
    bestpath = np.array([])
    for i in range(100):
        population, best, distance,resorve = _run(calculateDistance, constraints, lowwer, upper, pop_size, dim, mut_way, epochs, save, Q, k, CR)
        best1=np.append(best1,distance)
        if distance == np.min(best1):
            bestpath = best
        if distance <2380000:
            break
    # print("最优解：x=", best)
    # print("最优值：f(x)=", distance)
    print(calculateDistance(bestpath, dist_matrix1))
    toc2 = time.perf_counter()
    print(f"平均计算耗时:{(toc2 - tic1)/len(best1):0.4f}秒")
    print("平均距离",np.average(best1)/10000-0.002,"km")
    print("最小距离",np.min(best1)/10000-0.002,"km")
    print("最大距离",np.max(best1)/10000-0.002,"km")
    
    optimal_path = rearrange_path(bestpath, 115, 116)
    
    print("bestpath",optimal_path)
    plotPath(optimal_path,data_df)
    # import networkx as nx
    # from pylab import mpl
 
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    # mpl.rcParams['axes.unicode_minus']=False
    # G = nx.Graph()
    with open("coordinates.json", 'r', encoding='utf-8') as f1, \
        open("numtolocation.json", 'r', encoding='utf-8') as f2:
        data = json.load(f1)
        data1 = json.load(f2)
    optimal_path += 1
    lines_per_batch = 10  # 每批次打印的行数
    for i in range(0, len(optimal_path)):  
        print(data1[str(optimal_path[i])])  
    # i= 0
    # # for location, coords in data.items():
    # #     G.add_node(location[13:], pos=(coords[0], coords[1]))
    # #     print(location[13:], coords)


    # # for i in range(len(best_permutation) - 1):
    # #     G.add_edge(data1[str(best_permutation[i])][13:], data1[str(best_permutation[i + 1])][13:])

    # for location, coords in data.items():
    #     i+=1
    #     print(location[13:], coords)
    #     G.add_node(location[13:], pos=(coords[0], coords[1]))
        

    # for k in range(len(optimal_path)-1):
    #     G.add_edge(data1[str(optimal_path[k])][13:], data1[str(optimal_path[k+1])][13:])
    # pos=nx.get_node_attributes(G,'pos')
    # pos = nx.spring_layout(G)

    # # for location, coords in data.items():
    # #     i+=1
    # #     print(i, coords)
    # #     G.add_node(str(i), pos=(coords[0], coords[1]))
        


    # # for k in range(1,len(best_permutation)):
    # #     G.add_edge(str(k), str(k+1))
    # # pos = nx.spring_layout(G)

    # # 绘制图形
    # plt.figure(figsize=(12, 8))
    # nx.draw(G, pos, with_labels=True, node_size=50, font_size=4)
    # plt.savefig("tsp_path.png", format="png")
    # plt.show()