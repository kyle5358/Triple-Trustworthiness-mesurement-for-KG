EntityDict={}
sourcefile=open("entity2id.txt","r")
for line in sourcefile:
        list = line.split(" ")
        # print(list[0])
        if list[0] not in EntityDict.keys():
                EntityDict[list[0]] = list[1].strip('\n')
                # dict[list[0]]= [list[1].strip('\n')]
sourcefile.close()

# list_of_key = list(EntityDict.keys())
# list_of_value = list(EntityDict.values())
m=[]
n=[]
for key,value in EntityDict.items():
    if "OpExecutionMethod" in key:
        m.append(int(value))
    if "MMLCommand" in key:
        n.append(int(value))
print([min(m),max(m),min(n),max(n)])
