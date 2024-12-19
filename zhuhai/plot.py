import matplotlib.pyplot as plt
def plotPath(optimal_policy,node_file):
    pointsX = []
    pointsY = []
    for i in range(len(optimal_policy)-1):
        Node = optimal_policy[i]
        NodeX = node_file.loc[Node, 'longitude']
        NodeY = node_file.loc[Node, 'latitude']
        pointsX.append(NodeX)
        pointsY.append(NodeY)
    pointsX.append(node_file.loc[optimal_policy[0], 'longitude'])
    pointsY.append(node_file.loc[optimal_policy[0], 'latitude'])
    # fig, ax = plt.subplots(figsize=(20, 15), dpi=100)
    ax = plt.scatter(pointsX, pointsY, color = 'red',s=20)
    ax = plt.plot(pointsX, pointsY, color = 'blue')
    plt.ylim(22, 22.5)
    plt.xlim(113, 113.75)
    # for i in range(len(node_file)):
    #     locX = node_file.loc[i, 'longitude'] + 0.25
    #     locY = node_file.loc[i, 'latitude'] + 0.25
    #     label = i
    #     plt.text(locX, locY, str(label), family='serif', style='italic', fontsize=15, verticalalignment="bottom", ha='left', color='k')
    plt.savefig(r"jieguo.png")
    plt.show()
    plt.close()  # 就是这里 一定要关闭