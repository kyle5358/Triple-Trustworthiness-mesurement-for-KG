#将实体转换成id
import xlrd        # 读取excel文件需要的库
import glob
from tqdm import tqdm

fileList = glob.glob(r"C:\python_program\TTMFDataProcess\Positive\GenerateDataset\huawei\nodes\*.xlsx")  # 图片所在文件夹的路径
sqlfile = open("entity2id.txt", "w")  # 文件读写方式是覆盖
# 打开文件
m = 1
dict={}
for item in tqdm(fileList):
        start=m
        img_name = item.split('\\')[-1]  # excel名字
        print(img_name)
        data = xlrd.open_workbook(item)  # 旧版xlrd
# data = xlrd.open_workbook("AlarmCategory.xlsx")    # 旧版xlrd
        table = data.sheets()[0]  # 表头，第几个sheet表-1
        nrows = table.nrows  # 行数
        ncols = table.ncols  # 列数

        for ronum in range(1, nrows):
                row = table.cell_value(rowx=ronum, colx=1)  # 只需要修改你要读取的列数-1
                # values = strs(row)  # 调用函数，将行数据拼接成字符串
                sqlfile.write(row + " " + str(m) + '\n')
                m = m + 1
        end=m-1
        dict[img_name]=start
sqlfile.close()  # 关闭写入的文件
print(dict)