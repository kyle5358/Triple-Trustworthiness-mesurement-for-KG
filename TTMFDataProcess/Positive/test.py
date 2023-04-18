# for i in range(0,10):
#     print(i)

# RelDict={}

#


relfile = open("relation2id_old.txt", "r")
with open("relation2id.txt",'w') as f:
        for line in relfile:
                list = line.split(" ")
                m=int(list[1].strip('\n'))+1
                f.write(list[0] + " " + str(m) + '\n')
                # print(list[0])
                # list[1]
relfile.close()



