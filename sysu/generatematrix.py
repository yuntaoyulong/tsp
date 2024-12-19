from PIL import Image
import os  
import json
import cv2  
import pytesseract
from paddleocr import PaddleOCR
import sys  
import numpy as np
from math import radians, sin, cos, sqrt, atan2

ocr = PaddleOCR(use_angle_cls=True, use_gpu=True)  # 使用GPU预加载
# 设置 Tesseract 的路径（仅在必要时，视你的安装情况而定）
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 进行 OCR 识别，并获取识别结果的详细信息

# 指定要遍历的文件夹路径  
folder_path = "E:\\MiniProject\\sysu"
coordinates_dict = {}
numtolocation = {}
file_counter = 1
# 遍历文件夹  
x_start1 = 45
y_start1 = 2340
width1 = 1260
height1 = 220
x_start = 338
y_start = 1341
width = 581
height = 58
start = {}
end = {}
file_total = 150
textL = {}

# 保存当前的stdout  
original_stdout = sys.stdout  

def string_to_tuple(s):
    # 使用逗号分割字符串
    parts = s.split(',')
    # 将分割后的部分转换为浮点数，并放入元组中
    return (float(parts[0]), float(parts[1]))

# Haversine 公式计算两点之间的距离
def haversine(coord1, coord2):
    R = 6378.388  # 地球半径，单位为公里

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c *10000  # 乘以 10000，以便在整数范围内进行计算
    return int(distance)

# 打开文件以写入  
with open("E:\\MiniProject\\huishououtput.txt", 'w') as f:  
    sys.stdout = f  # 将stdout重定向到文件  

    for root, dirs, files in os.walk(folder_path):  
        for file in files:  
            jpg_path = os.path.join(root, file)
            print(os.path.join(root, file))  # 打印文件的完整路径
            if jpg_path.endswith('.jpg'):
                image_path = jpg_path
                image = cv2.imread(image_path)
                
                roi = image[y_start1:y_start1 + height1, x_start1:x_start1 + width1]
                roi1 = image[y_start:y_start + height, x_start:x_start + width]
                cv2.imwrite(f"E:\\MiniProject\\huishou1\\{file_counter}.jpg", roi)
                cv2.imwrite(f"E:\\MiniProject\\huishou1\\ab{file_counter}.jpg", roi1)
                text_list = ocr.ocr(roi, cls=True)  # 打开图片文件
                ordi = ocr.ocr(roi1, cls=True)
                tzan = ''
                for t in text_list[0]:
                    
                    tzan+=t[1][0]
                    
                if "唐家湾站" in tzan.split('m', 1)[1]:
                    start = file_counter
                elif "珠海站" in tzan.split('m', 1)[1]:
                    end = file_counter
                text = pytesseract.image_to_string(roi1) 
                text = string_to_tuple(text.replace(" ", "").strip())
                tzano = tzan.split('m', 1)[1]
                if tzano in textL:
                    tzano = tzano + '1'
                else:    
                    pass
                #tzano = tzan.split('m', 1)[1].replace('）', ')')
                textL[tzano] = text
                numtolocation[str(file_counter)] = tzano
                print(tzan)
                file_counter += 1

    for key, value in textL.items():
        print(key, value)
    output_file = "coordinates.json"  # 替换为你想要保存的文件路径
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(textL, f, ensure_ascii=False, indent=4)
    print(start, end)
    with open("numtolocation.json", 'w', encoding='utf-8') as f:
        json.dump(numtolocation, f, ensure_ascii=False, indent=4)

# 重置stdout为原始值  
sys.stdout = original_stdout

# 加载保存的经纬度信息
with open("coordinates.json", 'r', encoding='utf-8') as f:
    coordinates = json.load(f)

# 获取所有点的名称
points = list(coordinates.keys())

# 初始化距离矩阵
n = len(points)
distance_matrix = np.zeros((n, n))
centroid_matrix = np.zeros((3, n))


# 计算距离矩阵
for i in range(n):
    
    for j in range(n):
        if i != j:
            coord1 = coordinates[points[i]]
            coord2 = coordinates[points[j]]
            centroid_matrix[0, i] = i + 1  # 保存数字
            centroid_matrix[1, i] = coord1[0]  # 保存横坐标
            centroid_matrix[2, i] = coord1[1]  # 保存纵坐标
            distance_matrix[i, j] = haversine(coord1, coord2)


# 保存距离矩阵为 CSV 文件
np.savetxt("distance_matrix.csv", distance_matrix, delimiter=",")
np.savetxt("centroid_matrix.csv", centroid_matrix, delimiter=",")
print("距离矩阵已保存为 distance_matrix.csv")