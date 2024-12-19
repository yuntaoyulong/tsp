import numpy as np
import matplotlib.pyplot as plt

import random
from numba import njit

@njit(cache =True)
def select_center(first_center, data, k):
    centers = [first_center]
    for _ in range(1, k):
        assigned_data = assignment(data, centers)
        sum_closest_d = np.sum(assigned_data[:, -1])
        probabilities = assigned_data[:, -1] / sum_closest_d
        cumulative_probabilities = np.cumsum(probabilities)
        r = random.random()
        index = 0
        for prob in cumulative_probabilities:
            if prob > r:
                break
            index += 1
        centers.append(data[index, :2])
    return centers

@njit(cache =True)
def assignment(data, centers):
    num_points = data.shape[0]
    num_centers = len(centers)
    distances = np.empty((num_points, num_centers))
    for i in range(num_centers):
        dx = data[:, 0] - centers[i][0]
        dy = data[:, 1] - centers[i][1]
        distances[:, i] = np.sqrt(dx * dx + dy * dy)
    closest = np.zeros(num_points, dtype=np.int64)
    closest_d = np.empty(num_points)
    for j in range(num_points):
        min_dist = distances[j, 0]
        min_index = 0
        for i in range(1, num_centers):
            if distances[j, i] < min_dist:
                min_dist = distances[j, i]
                min_index = i
        closest[j] = min_index
        closest_d[j] = min_dist
    assigned_data = np.column_stack((data, closest, closest_d))
    return assigned_data

@njit(cache =True)
def update(assigned_data, centers, k):
    for i in range(k):
        mask = assigned_data[:, 2] == i
        points = assigned_data[mask][:, :2]
        if points.shape[0] > 0:
            centers[i][0] = np.mean(points[:, 0])
            centers[i][1] = np.mean(points[:, 1])
    return centers



@njit(cache =True)
def vector_product(vectorA, vectorB):
    """计算 x_1 * y_2 - x_2 * y_1"""
    return vectorA[0] * vectorB[1] - vectorA[1] * vectorB[0]

@njit(cache =True)
def length(pointA, pointB):
    """计算两点间距离"""
    dx = pointA[0] - pointB[0]
    dy = pointA[1] - pointB[1]
    return np.sqrt(dx * dx + dy * dy)

@njit(cache =True)
def check(A, B, C, D):
    """判断两线段是否相交"""
    AC = C - A
    AD = D - A
    BC = C - B
    BD = D - B
    CA = -AC
    CB = -BC
    DA = -AD
    DB = -BD

    cond1 = vector_product(AC, AD) * vector_product(BC, BD) <= 0
    cond2 = vector_product(CA, CB) * vector_product(DA, DB) <= 0
    return cond1 and cond2

@njit(cache =True)
def is_intersected(A, B, C, D):
    """检查三种组合的相交情况"""
    ch1 = check(A, B, C, D)
    ch2 = check(A, C, B, D)
    ch3 = check(A, D, B, C)
    return ch1, ch2, ch3

@njit(cache =True)
def check_intersections(points):
    n = len(points)
    intersections = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    ch1, ch2, ch3 = is_intersected(points[i], points[j], points[k], points[l])
                    if ch1:
                        if length(points[i], points[j]) > length(points[k], points[l]):
                            intersections[i, j] = 1
                            intersections[j, i] = 1
                        else:
                            intersections[k, l] = 1
                            intersections[l, k] = 1
                    if ch2:
                        if length(points[i], points[k]) > length(points[j], points[l]):
                            intersections[i, k] = 1
                            intersections[k, i] = 1
                        else:
                            intersections[j, l] = 1
                            intersections[l, j] = 1
                    if ch3:
                        if length(points[i], points[l]) > length(points[j], points[k]):
                            intersections[i, l] = 1
                            intersections[l, i] = 1
                        else:
                            intersections[j, k] = 1
                            intersections[k, j] = 1
        intersections[i, i] = 1
    return intersections
@njit(cache =True)
def kmeans(k,data):
    
    first_center_index = random.randint(0, data.shape[0] - 1)
    first_center = data[first_center_index, :2]
    centers = select_center(first_center, data, k)
    
    for _ in range(10):
        assigned_data = assignment(data, centers)
        centers = update(assigned_data, centers, k)
    
    assigned_data = assignment(data, centers)
    color_indices = assigned_data[:, 2].astype(np.int64)
    # 将中心点坐标转换为 numpy 数组
    points = centers  # 假设 centers 是中心点坐标的列表

    # 计算交叉关系矩阵
    intersections = np.ones((k+1, k+1), dtype=np.int64)
    
    intersectionspre = check_intersections(points)

    
    intersections[:k,:k] = intersectionspre

    intersections[k,color_indices[116]]= 0
    intersections[k,color_indices[115]]= 0
    intersections[color_indices[116],k]= 0
    intersections[color_indices[115],k]= 0
    # 创建全部元素的集合
    # reachable1 = np.zeros((k+1,k+1), dtype=np.int64)
    
    # reachable1[k][color_indices[116]]= 1
    # reachable1[k][color_indices[115]]= 1
    # reachable1[color_indices[116]][k]= 1
    # reachable1[color_indices[115]][k]= 1
    reachable1 = np.logical_not(intersections)  
 
# 获取每个中心点的可达集合的索引  
    reachable = [np.where(reachable1[i])[0] for i in range(k+1)]  

    # 假设 color_indices 是数据点的聚类标签数组
    # 将特殊索引的元素添加到 reachable 中（请根据实际情况调整索引）
    # reachable.append(np.array([color_indices[115], color_indices[116]]))
    # reachable[color_indices[115]] = np.append(reachable[color_indices[115]], k)
    # reachable[color_indices[116]] = np.append(reachable[color_indices[116]], k)

    # 更新 color_indices 数组
    
    # plt.scatter(data[:, 0], data[:, 1], color=[color_list[i] for i in color_indices], alpha=0.5, edgecolor='b')
    # for i, center in enumerate(centers):
    #     plt.scatter(center[0], center[1], color=color_list[i], linewidths=6)
    #     plt.text(center[0], center[1], f'Class {i}', fontsize=12, ha='right')
    # plt.xlim(22, 22.5)
    # plt.ylim(113, 113.75)
    # plt.show()
    # print(color_indices, reachable)
    color_indices = np.append(color_indices, k)
    return color_indices, reachable

# if __name__ == '__main__':
#     import json
#     location_path = "coordinates.json"
#     with open(location_path, 'r', encoding='utf-8') as f1:
#         data_json = json.load(f1)
#     data = np.array(list(data_json.values()), dtype=np.float64)
    
#     print(kmeans(8,data))