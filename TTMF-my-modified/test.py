# from search import ReadAllTriples
# from SearchPaths2 import searchpath
# file_test = "test.txt"
#
# dict = ReadAllTriples([file_test])
#
#
# Paths = {}
# pathlist = []
# startnode = 'A'
# taillist = [startnode]
# Paths = searchpath(startnode, startnode, dict, taillist, Paths, pathlist, 4)
# for head in Paths.keys():
#     print(len(Paths[head]))
# print(Paths)
#
# file = open(file_test ,"r")
# line_dict = {}  # 正负都有
# headlist = []  # 正负都有
# # 头实体，尾实体之间的所有路径
# for linet in file:
#     list = linet.rstrip('\n').split(' ')
#
#     if list[0] + '_' + list[1] in line_dict.keys():
#         if (list[0], list[1], list[2]) not in line_dict[list[0] + '_' + list[1]]:
#             line_dict[list[0] + '_' + list[1]].append((list[0], list[1], list[2]))
#     else:
#         line_dict[list[0] + '_' + list[1]] = [(list[0], list[1], list[2])]
#     if list[0] not in headlist:
#         headlist.append(list[0])
#
# file.close()
# print(line_dict)
# print(headlist)
#
# path_num_list={}
# path_num=len(Paths[head])
# if path_num in path_num_list.keys():
#     path_num_list[path_num]+=1
# else:
#     path_num_list[path_num]=1
#
# print(path_num_list)

# # m={'1':[2,3,4],'2':3}
# # print(len(m['1']))
# import matplotlib.pyplot as plt
# data={1: 80359, 2: 21264, 3: 1065, 12: 56, 8: 184, 24: 6, 4: 1112, 9: 86, 6: 268, 5: 395, 7: 146, 10: 52, 19: 12, 15: 44, 13: 18, 31: 10, 11: 124, 29: 17, 16: 27, 22: 5, 23: 18, 20: 22, 35: 8, 14: 30, 17: 21, 53: 1, 27: 6, 18: 6, 25:5, 37: 3, 21: 11, 32: 6, 34: 9, 77: 2, 200: 1, 635: 2, 26: 3, 36: 4, 30: 10, 89: 2, 191: 2, 44: 2, 28: 5, 75: 3,118: 2, 141: 3, 131: 1, 366: 1, 107: 1, 222: 1, 40: 3, 166: 1, 115: 1, 71: 1, 70: 1, 59: 1, 67: 4, 43: 3, 41: 4, 142: 1, 113: 1, 108: 1, 74: 2, 80: 1, 84: 1, 100: 1, 110: 2, 66: 2, 72: 2, 57: 1, 48: 6, 83: 1, 39: 3, 54: 1, 226: 1, 42: 1, 163: 1, 64: 1, 49: 1, 90: 1, 61: 1, 47: 1, 86: 1, 38: 7, 87: 6, 111: 1, 33: 3, 76: 1, 170: 1, 46: 1, 73:  1, 157: 1}
#
# # names = list(data.keys())
# # values = list(data.values())
# #
# # plt.bar(names, values)
# # plt.xlim(0, 70)
# # plt.show()
# total = sum(data.values())
# print(total)
trainfile="test.txt"

# trainfile2="test_new2.txt"
# from ResourceRankConfidence import get_data_txt
# trainExamples, confidence = get_data_txt(trainfile)
# trainExamples2,_=get_data_txt(trainfile2)
# print(trainExamples)
# trainExamples.append(trainExamples2)
# print(trainExamples)
# from search import ReadAllTriples
# print(ReadAllTriples([trainfile]))

# def ReadAllTriples(files):
#     dict = {}
#
#     for f in files:
#         file = open(f, "r")
#         for line in file:
#             list = line.split(" ")
#
#             if list[0] in dict.keys():
#                 if list[1] in dict.get(list[0]).keys():
#                     dict.get(list[0]).get(list[1]).append(list[2].strip('\n'))
#                 else:
#                     dict.get(list[0])[list[1]] = [list[2].strip('\n')]
#             else:
#                 dict[list[0]] = {list[1]:[list[2].strip('\n')]}
#
#         # for key in dict.keys():
#         #     print(key+' : ',dict[k])
#         file.close()
#
#     return dict
#
# print(ReadAllTriples([trainfile]))
import time
from pygraph.classes.digraph import digraph
# import digraph
import os
from search import ReadAllTriples,DFS
# file_data = ""
file_entity = "entity2id.txt"
# file_train = file_data + "/train.txt"
# file_test = file_data + "/test.txt"
# file_valid = file_data + "/valid.txt"
# file_train = file_data + "/conf_train2id.txt"
# file_test = file_data + "/conf_test2id.txt"
# file_valid = file_data + "/conf_valid2id.txt"
file_subGraphs =  "D:\\1python_program\\TTMF-my - modified\\subGraphs_4\\"

dict = ReadAllTriples([trainfile])
print("dict size--", dict.__len__())
print("ReadAllTriples is done!")

file = open(file_entity, "r")

for line in file:
    list = line.split(" ")
    node0 = list[1].strip('\n')
    print("node0-----", node0)

    dg = digraph()
    dg.add_node(node0)
    t1 = time.perf_counter()
    dg = DFS(dict, dg, node0, depth=4)

    fo = open(file_subGraphs + node0 + ".txt", "w")
    NODE = ""
    for nodei in dg.nodes():
        NODE = NODE + nodei + "\t"
    fo.write(NODE + '\n')

    for e in dg.edges():
        fo.write(e[0] + "\t" + e[1] + "\t" + str(dg.edge_weight(e)) + '\n')
    fo.close()

    t2 = time.perf_counter()
    # time.sleep(1)
    print(t2 - t1)
    # print(dg.nodes().__len__())
    # for edge in dg.edges():
    #     print('edge----',edge)
file.close()

# class PRIterator:
#     __doc__ = '''计算一张图中的PR值'''
#
#     def __init__(self, dg, core_node):
#         self.damping_factor = 0.85  # 阻尼系数,即α
#         self.max_iterations = 500  # 最大迭代次数
#         self.min_delta = 0.00001  # 确定迭代是否结束的参数,即ϵ
#         self.core_node = core_node
#         self.graph = dg
#
#     def page_rank(self):
#         print('******')
#         cout =0
#          # 先将图中没有出链的节点改为对所有节点都有出链
#         for node in self.graph.nodes():
#             if len(self.graph.neighbors(node)) == 0:
#                 cout +=1
#                 # print(cout)
#                 digraph.add_edge(self.graph, (node, node), wt=0.5)
#                 if not digraph.has_edge(self.graph, (node, core_node)):
#                   digraph.add_edge(self.graph, (node, core_node), wt=0.5)
#                 # for node2 in self.graph.nodes():
#                 #     # print('$$$$$$$')
#                 #     if node !=node2:
#                 #         digraph.add_edge(self.graph, (node, node2),wt=float(1/len(self.graph.nodes())))
#
#         print(cout)
#
#         nodes = self.graph.nodes()
#         graph_size = len(nodes)
#
#         if graph_size == 0:
#             return {}
#
#         # page_rank = dict.fromkeys(nodes, 1.0 / graph_size)  # 给每个节点赋予初始的PR值
#         page_rank = dict.fromkeys(nodes, 0.0)  # 给每个节点赋予初始的PR值
#         page_rank[core_node] = 1.0
#         # print(page_rank)
#         damping_value = (1.0 - self.damping_factor) / graph_size  # 公式中的(1−α)/N部分
#         print('start iterating...')
#         flag = False
#         for i in range(self.max_iterations):
#             change = 0
#             for node in nodes:
#                 rank = 0
#                 for incident_page in self.graph.incidents(node):  # 遍历所有“入射”的页面
#                     # count = 0
#                     # for neighboredge in self.graph.neighbors(incident_page):
#                     #     count += self.graph.edge_weight((incident_page,neighboredge))
#                     # rank += self.damping_factor * (page_rank[incident_page] / count * self.graph.edge_weight((incident_page,node)))
#                     # rank += self.damping_factor * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))
#
#                     rank += self.damping_factor * page_rank[incident_page] * float(self.graph.edge_weight((incident_page,node)))
#                 rank += damping_value
#                 change += abs(page_rank[node] - rank)  # 绝对值
#                 page_rank[node] = rank
#
#             # print("This is NO.%s iteration" % (i + 1))
#             # print(page_rank)
#
#             if change < self.min_delta:
#                 flag = True
#                 # print("\n\nfinished in %s iterations!" % i)
#                 break
#         if flag == False:
#             print("finished out of %s iterations!" % self.max_iterations)
#         return page_rank
#
# file_entityRank = "D:\\1python_program\\TTMF-my - modified\\entityRank_4\\"
# file_subGraphs =  "D:\\1python_program\\TTMF-my - modified\\subGraphs_4\\"
#
# #用子图找到entityrank_4
# for files in os.listdir(file_subGraphs):
#     file = open(file_subGraphs+files, "r")
#     dg = digraph()
#     core_node = os.path.splitext(files)[0]
#     print('corenode----',core_node)
#     for i, line in enumerate(file):
#         if i == 0:
#             list = line.rstrip('\t\n').rstrip('\t').rstrip('\n').split("\t")
#             for n in list:
#                 dg.add_node(n.strip("\t").strip("\n"))
#         else:
#
#             list = line.split("\t")
#             # dg.add_edge((list[0], list[1]), wt=int(list[2].strip("\n")))
#             dg.add_edge((list[0], list[1]), wt=list[2].strip("\n"))
#     print('dg size...', dg.nodes().__len__())
#
#     pr = PRIterator(dg, core_node)
#     page_ranks = pr.page_rank()
#
#     # print("The final page rank is\n", page_ranks)
#     fo = open(file_entityRank + core_node + ".txt", "w")
#     for key in page_ranks.keys():
#         fo.write(key +"\t" + str(page_ranks.get(key)) + "\n")
#     fo.close()
#     file.close()