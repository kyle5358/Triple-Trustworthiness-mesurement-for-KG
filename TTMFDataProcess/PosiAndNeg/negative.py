
#根据每个关系三元组txt，打乱后，替换头或尾实体，放入train, test, valid
import glob
from tqdm import tqdm
# imageList = glob.glob(r"C:\python_program\TTMF_my\test\*.xlsx")  # 图片所在文件夹的路径
imageList = glob.glob(r"C:\python_program\TTMF_my\test\*.txt")  # 图片所在文件夹的路径
result_dir="C:\\python_program\\TTMF_my\\result\\"

# f=open("C:\\python_program\\TTMF_my\\result\\")

import random
import math

# lines = []
# with open("C:\\python_program\\TTMF_my\\conf\\ACHIEVE_BY.txt", 'r', encoding='utf-8') as f:  # 需要打乱的原文件位置
#     for line in f:
#         lines.append(line)
# random.shuffle(lines)
#
# train_num=math.ceil(len(lines)*0.8)
# test_num=math.ceil(len(lines)*0.1)
# valid_num=len(lines)-train_num-test_num
# print([train_num,test_num,valid_num])
#
# for i in range(0,train_num):
#     list = lines[i].split(" ")
#
#     train_file.write(list[0]+" "+list[1]+" "+list[2].strip('\n')+" "+"1" + '\n')
#     if i%2==0:
#         list[0]=random.randint(13359,14246)
#     else:
#         list[1]=random.randint(21105, 21200)
#     train_file.write(str(list[0])+" "+str(list[1])+" "+list[2].strip('\n')+" "+"-1" + '\n')


def write_dataset(file_path,array):
    a,b,c,d=array
    # print([a,b,c,d])
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:  # 需要打乱的原文件位置
        for line in f:
            lines.append(line)
    random.shuffle(lines)

    train_num = math.ceil(len(lines) * 0.8)
    test_num = math.ceil(len(lines) * 0.1)
    valid_num = len(lines) - train_num - test_num
    # print([train_num, test_num, valid_num])

    train_file = open("C:\\python_program\\TTMF_my\\positive\\train.txt", 'a', encoding='utf-8')  # 输出文件位置
    for i in range(0, train_num):
        list = lines[i].split(" ")
        train_file.write(list[0] + "\t" + list[1] + "\t" + list[2].strip('\n')  + '\n')
    #     if i % 2 == 0:
    #         list[0] = random.randint(a, b)
    #     else:
    #         list[1] = random.randint(c, d)
    #     train_file.write(str(list[0]) + " " + str(list[1]) + " " + list[2].strip('\n') + " " + "-1" + '\n')
    train_file.close()

    test_file = open("C:\\python_program\\TTMF_my\\positive\\test.txt", 'a', encoding='utf-8')  # 输出文件位置
    for i in range(train_num,train_num+test_num):
        list = lines[i].split(" ")
        test_file.write(list[0] + "\t" + list[1] + "\t" + list[2].strip('\n')  + '\n')
        # if i % 2 == 0:
        #     list[0] = random.randint(a, b)
        # else:
        #     list[1] = random.randint(c, d)
        # test_file.write(str(list[0]) + " " + str(list[1]) + " " + list[2].strip('\n') + " " + "-1" + '\n')
    test_file.close()

    valid_file = open("C:\\python_program\\TTMF_my\\positive\\valid.txt", 'a', encoding='utf-8')  # 输出文件位置
    for i in range(train_num+test_num,train_num+test_num+valid_num):
        list = lines[i].split(" ")
        valid_file.write(list[0] + "\t" + list[1] + "\t" + list[2].strip('\n')  + '\n')
        # if i % 2 == 0:
        #     list[0] = random.randint(a, b)
        # else:
        #     list[1] = random.randint(c, d)
        # valid_file.write(str(list[0]) + " " + str(list[1]) + " " + list[2].strip('\n') + " " + "-1" + '\n')
    valid_file.close()

array=[
[13359,14416,21105,21200],
[12142,13268,410,11364],
[18459, 20819, 21207, 34896],
[18459, 20819, 21207, 34896],
[13359, 14416, 12142, 13268],
[1, 351, 342, 351],
[410, 11364, 10981, 11364],
[12142, 13268, 13255, 13268],
[14417, 14616, 14601, 14616],
[1, 351, 1, 351],
[11365, 12141, 11365, 12141],
[13359, 14416, 13359, 14416],
[14417, 14616, 410, 11364],
[20854, 20910, 13359, 14416],
[13359, 14416, 13359, 14416],
[18459, 20819, 18459, 20819],
[13359, 14416, 18459, 20819],
[13359, 14416, 21098, 21104],
[13314, 13327, 13328, 13358],
[18459, 20819, 18459, 20819],
[18459, 20819, 1, 351],
[18459, 20819, 18459, 20819],
[18459, 20819, 18459, 20819],
[13359, 14416, 20854, 20910],
[18459, 20819, 10981, 11364],
[13359, 14416, 21067, 21080],
[13359, 14416, 21081, 21097],
[13314, 13327, 13359, 14416],
[13359, 14416, 13359, 14416],
[13269, 13309, 352, 409],
[13314, 13327, 13269, 13309],
[18459, 20819, 14417, 14616],
[12142, 13268, 21207, 34896],
[13310, 13311, 14417, 14616],
[13359, 14416, 20911, 21066],
[14247, 14416, 20926, 21066],
[13359, 14416, 21207, 34896],
[13269, 13309, 13359, 14416],
[21105, 21200, 14617, 18458]
]
filePath="C:\\python_program\\TTMF_my\\GenerateDataset\\conf\\"
# for item in tqdm(fileList):
# write_dataset(filePath+"ACHIEVE_BY.txt",13359,14246,21105, 21200)
write_dataset(filePath+"ACHIEVE_BY.txt",array[0])
write_dataset(filePath+"ASSOCIATE_WITH.txt",array[1])
write_dataset(filePath+"ATTRIBUTE_dynamic.txt",array[2])
write_dataset(filePath+"ATTRIBUTE_static.txt",array[3])
write_dataset(filePath+"BASED_ON.txt",array[4])
write_dataset(filePath+"BELONG_alarm.txt",array[5])
write_dataset(filePath+"BELONG_counter.txt",array[6])
write_dataset(filePath+"BELONG_feature.txt",array[7])
write_dataset(filePath+"BELONG_kpi.txt",array[8])
write_dataset(filePath+"CAUSE.txt",array[9])
write_dataset(filePath+"CHECK_NEXT.txt",array[10])
write_dataset(filePath+"CONFLICT_WITH.txt",array[11])
write_dataset(filePath+"CONSIST_OF.txt",array[12])
write_dataset(filePath+"CORRESPOND_WITH.txt",array[13])
write_dataset(filePath+"DEPEND_ON.txt",array[14])
write_dataset(filePath+"ENTITY_OF.txt",array[15])
write_dataset(filePath+"EXECUTED_BY.txt",array[16])
write_dataset(filePath+"EXECUTED_IF.txt",array[17])
write_dataset(filePath+"FULFILL.txt",array[18])
write_dataset(filePath+"HAS.txt",array[19])
write_dataset(filePath+"HAS_ALARM.txt",array[20])
write_dataset(filePath+"HAS_CHILDMO.txt",array[21])
write_dataset(filePath+"HAS_CHILDMO_dynamic.txt",array[22])
write_dataset(filePath+"HAS_CONFLICT.txt",array[23])
write_dataset(filePath+"HAS_COUNTERSET.txt",array[24])
write_dataset(filePath+"HAS_DEPENDENCY.txt",array[25])
write_dataset(filePath+"HAS_IMPACT.txt",array[26])
write_dataset(filePath+"HAS_OPERATION.txt",array[27])
write_dataset(filePath+"HAS_SUBOPERATION.txt",array[28])
write_dataset(filePath+"JUDGE_BY.txt",array[29])
write_dataset(filePath+"MATCH.txt",array[30])
write_dataset(filePath+"MEASURE.txt",array[31])
write_dataset(filePath+"POSSIBLE_SET.txt",array[32])
write_dataset(filePath+"RELATE.txt",array[33])
write_dataset(filePath+"SATISFY.txt",array[34])
write_dataset(filePath+"SATISFY_e.txt",array[35])
write_dataset(filePath+"SET.txt",array[36])
write_dataset(filePath+"TRIGGER.txt",array[37])
write_dataset(filePath+"USE.txt",array[38])




