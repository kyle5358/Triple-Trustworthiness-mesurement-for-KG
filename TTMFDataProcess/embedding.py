import json

with open("transe.json",encoding='utf-8') as a:
    result=json.load(a)
    # print(result)
    # ent_embed=result.get("ent_embeddings.weight")
    rel_embed=result.get("rel_embeddings.weight")
    ent_embed = result.get("ent_embeddings.weight")
    # print(ent_embed.type)
    # print(len(rel_embed[0]))
    # print(rel_embed[0])

    # print(len(rel_embed))
    # print(len(ent_embed))
    # print(len(ent_embed[0]))
    # print(ent_embed[0])

entityfile="D:\\1python_program\\TTMFDataProcess\\Positive\\entity2id.txt"
relationfile="D:\\1python_program\\TTMFDataProcess\\Positive\\relation2id.txt"

Entitylist=[]
sourcefile=open(entityfile,"r")
for line in sourcefile:
        list = line.split(" ")
        # print(list[0])
        if list[0] not in Entitylist:
                Entitylist.append(list[0])
                # dict[list[0]]= [list[1].strip('\n')]
sourcefile.close()

Relationlist=[]
relfile=open(relationfile,"r")
for line in relfile:
        list = line.split(" ")
        # print(list[0])
        if list[0] not in Relationlist:
                Relationlist.append(list[0])
                # dict[list[0]]= [list[1].strip('\n')]
relfile.close()
# print(Relationlist)

with open("Relation2vec.txt", 'w', encoding='UTF-8') as fp:
    for i in range(0,33):
        my_string = ""
        for j in rel_embed[i]:
            my_string += ' ' + str(j)
        fp.write(Relationlist[i]+my_string+ '\n')
print('rel2vec生成')

with open("Entity2vec.txt", 'w', encoding='UTF-8') as fp:
    for i in range(0,34896):
        my_string = ""
        for j in ent_embed[i]:
            my_string += ' ' + str(j)
        fp.write(Entitylist[i]+my_string+ '\n')
print('ent2vec生成')
