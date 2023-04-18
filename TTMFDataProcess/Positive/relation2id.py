# -*- coding:utf-8 -*-
import glob

imageList = glob.glob(r"C:\python_program\TTMFDataProcess\Positive\GenerateDataset\huawei\edges\*.xlsx")  # 图片所在文件夹的路径
f = open('relation2id.txt', 'w')  # 创建标签文件存放图片文件名
m=0
for item in imageList:
    # print(item)   # D:\0_me_python\目标检测\SSD-pytorch-main\SSD-pytorch-main\coco\val2017\000000000139.jpg
    # img_name = item.split('/')[-1]  # 图片文件名018.jpg
    img_name = item.split('\\')[-1]  # 图片文件名018.jpg
    relation=img_name.split('.')[0]
    f.write(relation + " "+str(m)+'\n')  # 将图片文件名逐行写入txt
    m=m+1
f.close()
print('OK')
