# -*- coding: utf-8 -*-
import os
import math
from pygraph.classes.digraph import digraph
# import numpy as np
from numpy import *

def get_data_txt(trainfile):
    train_triple = []
    train_confidence = []

    f = open(trainfile, "r")
    lines = f.readlines()
    for line in lines:
        tri = line.rstrip('\r\n').split(' ')
        train_triple.append((int(tri[0]), int(tri[1]), int(tri[2]), int(tri[3])))
        if tri[3] == '1':
            train_confidence.append([0, 1])
        else:
            train_confidence.append([1, 0])
    f.close()

    return train_triple, train_confidence

# def get_data_txt_new(trainfile):
#     train_triple = []
#     f = open(trainfile, "r")
#     lines = f.readlines()
#     for line in lines:
#         tri = line.rstrip('\r\n').rstrip('\n').rstrip('\r').split(' ')
#         train_triple.append((int(tri[0]), int(tri[1]), int(tri[2])))
#     f.close()
#     return train_triple

def get_dict_entityRank(entityRank):
    dict = {}

    files = os.listdir(entityRank)
    for file in files:
        # print(file)
        fo = open(entityRank + file, 'r')
        lines = fo.readlines()
        dict_l = {}
        for line in lines:

            nodes = line.rstrip('\n').split('\t')
            if nodes[0] == '':
                continue
            dict_l[int(nodes[0])] = float(nodes[1])
        dict[int(os.path.splitext(file)[0])] = dict_l
        fo.close()
    return dict

def getThreshold(rrank):

    distanceFlagList = rrank
    distanceFlagList = sorted(distanceFlagList, key=lambda sp: sp[0], reverse=True)

    threshold = distanceFlagList[0][0] + 0.01
    maxValue = 0
    currentValue = 0
    for i in range(1, len(distanceFlagList)):
        if distanceFlagList[i - 1][1] == 1:
            currentValue += 1
        else:
            currentValue -= 1

        if currentValue > maxValue:
            threshold = (distanceFlagList[i][0] + distanceFlagList[i - 1][0]) / 2.0
            maxValue = currentValue
    # print('threshold... ', threshold)
    return threshold

def rrcThreshold(tcDevExamples, dict_entityRank):
    threshold_dict = {}
    rrank_dict = {}
    for tuple in tcDevExamples:
        #print("实体2"+" "+str(tuple[1]))
        if tuple[1] in dict_entityRank[tuple[0]].keys():
            v = (dict_entityRank[tuple[0]][tuple[1]], tuple[3])#尾节点相对头节点的重要性，正样本还是负样本
        else:
            v = (0.0, tuple[3])

        if tuple[0] not in rrank_dict.keys():
            rrank_dict[tuple[0]] = [v]
        else:
            rrank_dict[tuple[0]].append(v)

    for it in rrank_dict.keys():
        threshold_dict[it] = getThreshold(rrank_dict[it])

    return threshold_dict

def get_RRankConfidence(threshold_dict, tcDevExamples, dict_entityRank):

    confidence_dict = []

    right = 0.0
    for triple in tcDevExamples:
        if triple[0] in threshold_dict.keys():
            threshold = threshold_dict[triple[0]]
        else:
            print('threshold is None !!!!!!!!')
            threshold = 0.5
        rankvalue = 0.0

        f = 0.001
        if triple[1] in dict_entityRank[triple[0]].keys():
            rankvalue = dict_entityRank[triple[0]][triple[1]]
            f = 1.0 / (1.0 + math.exp(-25 * (rankvalue - threshold)))

        confidence_dict.append(f)

        if rankvalue >= threshold and triple[3] == 1:
            right +=1.0

        elif rankvalue < threshold and triple[3] == -1:
            right += 1.0

    print('RRankConfidence accuracy ---- ', right / len(tcDevExamples))

    return confidence_dict

def get_f(head, tail, threshold_dict,dict_entityRank):

    if head in threshold_dict.keys():
        threshold = threshold_dict[head]
    else:
        print('threshold is None !!!!!!!!')
        threshold = 0.5

    f = 0.001
    if tail in dict_entityRank[head].keys():
        rankvalue = dict_entityRank[head][tail]
        try:
            f = 1.0 / (1.0 + math.exp(-25 * (rankvalue - threshold)))
        except OverflowError:
            f = 0.001
    return f

def get_features_2file(dict_entityRank, file_subGraphs, threshold_dict, file_RRfeatures):
    features = []
    classlabels = []

    for id in range(1, len(dict_entityRank)+1):
        fw = open(file_RRfeatures + str(id) + '.txt', 'w')
        print(id)

        core_node = str(id)
        print('corenode----', core_node)

        file = open(file_subGraphs + core_node + '.txt', "r")
        dg = digraph()
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                list = line.rstrip('\t\n').rstrip('\t').rstrip('\n').split("\t")
                for n in list:
                    dg.add_node(n.strip("\t").strip("\n"))
            else:
                list = line.rstrip('\n').split("\t")
                # dg.add_edge((list[0], list[1]), wt=int(list[2].strip("\n")))
                dg.add_edge((list[0], list[1]))
        # print('dg size...', dg.nodes().__len__())

        rudu = {}
        chudu = {}
        for d in dg.nodes():
            rudu[d] = len(dg.incidents(d))
            chudu[d] = len(dg.neighbors(d))

        depdict = {core_node: 0}
        list = dg.neighbors(core_node)
        list2 = [0] * len(list)
        # i = 0
        while list.__len__()>0:
            node = list[0]
            # print(node)
            if node not in depdict.keys():

                depdict[node] = list2[0]+1

                for node2 in dg.neighbors(node):
                    if node2 not in depdict.keys():
                        list.append(node2)
                        list2.append(depdict[node])
            del list[0]
            del list2[0]

        for d in dg.nodes():

            # rr = dict_entityRank[int(core_node)][int(d)]
            rr = get_f(int(core_node), int(d), threshold_dict, dict_entityRank)#rr是论文中的由pagerank得到的，尾节点从头节点获取的资源

            depth = depdict[d]
            print(d, rr, rudu[core_node], chudu[core_node], rudu[d], chudu[d], depth)

            fw.write(d + '\t' + str(rr) +'\t'+
                     str(rudu[core_node]) +'\t'+ str(chudu[core_node]) +'\t'+
                     str(rudu[d]) +'\t'+ str(chudu[d]) +'\t'+
                     str(depth) + '\n')
        fw.close()

def get_dict_features(file_RRfeatures):
    dict = {}

    files = os.listdir(file_RRfeatures)
    for file in files:
        print(file)
        fo = open(file_RRfeatures + file, 'r')
        lines = fo.readlines()
        dict_l = {}
        for line in lines:

            nodes = line.rstrip('\n').split('\t')

            dict_l[int(nodes[0])] = [float(nodes[1]), float(nodes[2]), float(nodes[3]), float(nodes[4])]
        dict[int(os.path.splitext(file)[0])] = dict_l
        fo.close()
    return dict

if __name__ == "__main__":

    file_data = "/data1/yk/Lab/TTMF_remove_gather/dataset/"
    # # file_entityRank = file_data + "/ResourceRank_4/"
    file_subGraphs = file_data + "/subGraphs_4/"
    trainfile = file_data + "/conf_train2id.txt"
    testfile=file_data+"/conf_test2id_new.txt"
    # validfile=file_data+"/conf_valid2id.txt"

    entityRank = file_data + "/entityRank_4/"
    file_RRfeatures = file_data + "/ResourceRank_4/"

    print('start...')

    dict_entityRank = get_dict_entityRank(entityRank)
    print(dict_entityRank.__len__())
    # dict_features = get_dict_features(file_RRfeatures)
    # print(dict_features.__len__())

    trainExamples, confidence = get_data_txt(trainfile)
    testExamples,_=get_data_txt(testfile)
    # validExamples,_=get_data_txt(validfile)
    trainExamples.extend(testExamples)
    # trainExamples.extend(validExamples)

    threshold_dict = rrcThreshold(trainExamples, dict_entityRank)
    get_features_2file(dict_entityRank, file_subGraphs, threshold_dict, file_RRfeatures)