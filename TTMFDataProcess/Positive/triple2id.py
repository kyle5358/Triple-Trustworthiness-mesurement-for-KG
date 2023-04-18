
import xlrd        # 读取excel文件需要的库
import glob
        # img_name = item.split('\\')[-1]  # excel名字
        # print(img_name)
EntityDict={}
sourcefile=open("entity2id.txt","r")
for line in sourcefile:
        list = line.split(" ")
        # print(list[0])
        if list[0] not in EntityDict.keys():
                EntityDict[list[0]] = list[1].strip('\n')
                # dict[list[0]]= [list[1].strip('\n')]
sourcefile.close()

RelDict={}
relfile=open("relation2id.txt","r")
for line in relfile:
        list = line.split(" ")
        # print(list[0])
        if list[0] not in RelDict.keys():
                RelDict[list[0]] = list[1].strip('\n')
                # dict[list[0]]= [list[1].strip('\n')]
relfile.close()
print(RelDict)

import glob
from tqdm import tqdm
# imageList = glob.glob(r"C:\python_program\TTMF_my\test\*.xlsx")  # 图片所在文件夹的路径
imageList = glob.glob(r"C:\python_program\TTMFDataProcess\Positive\GenerateDataset\huawei\edges\*.xlsx")  # 图片所在文件夹的路径
result_dir="C:\python_program\TTMFDataProcess\Positive\\triple\\"
# f = open('relation2id.txt', 'a')  # 创建标签文件存放图片文件名

for item in tqdm(imageList):
    # print(item)   # D:\0_me_python\目标检测\SSD-pytorch-main\SSD-pytorch-main\coco\val2017\000000000139.jpg

    img_name = item.split('\\')[-1]  # 图片文件名018.jpg
    relation=img_name.split('.')[0]
    text_name=result_dir+relation+".txt"
    f=open(text_name,'w')
    data = xlrd.open_workbook(item)  # 旧版xlrd
    # data = xlrd.open_workbook("AlarmCategory.xlsx")    # 旧版xlrd
    table = data.sheets()[0]  # 表头，第几个sheet表-1
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数

    for ronum in range(1, nrows):
            entity1 = table.cell_value(ronum, 0)
            entity2 = table.cell_value(ronum, 1)
            relation = table.cell_value(ronum, 2)

            entity1id = EntityDict[entity1]
            entity2id = EntityDict[entity2]
            relation2id = RelDict[relation]
            # values = strs(row)  # 调用函数，将行数据拼接成字符串
            f.write(entity1id + " " + entity2id + " " + relation2id + '\n')
    f.close()



# #先生成每个标题.txt，再打乱选train, test, valid, 最后生成负样本
# sqlfile = open("triple2id.txt", "a")  # 文件读写方式是追加
#
# data = xlrd.open_workbook(r"C:\python_program\TTMF_my\huawei\edges\ACHIEVE_BY.xlsx")  # 旧版xlrd
# # data = xlrd.open_workbook("AlarmCategory.xlsx")    # 旧版xlrd
# table = data.sheets()[0]  # 表头，第几个sheet表-1
# nrows = table.nrows  # 行数
# ncols = table.ncols  # 列数
#
# for ronum in range(1, nrows):
#         entity1=table.cell_value(ronum, 0)
#         entity2=table.cell_value(ronum, 1)
#         relation=table.cell_value(ronum, 2)
#
#         entity1id=EntityDict[entity1]
#         entity2id = EntityDict[entity2]
#         relation2id = RelDict[relation]
#         # values = strs(row)  # 调用函数，将行数据拼接成字符串
#         sqlfile.write(entity1id + " " + entity2id + " " +relation2id +'\n')
#
# sqlfile.close()  # 关闭写入的文件

