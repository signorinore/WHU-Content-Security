import torch

pthfile = r'../weights/lprnet-pretrain.pth'  # .pth文件的路径
model = torch.load(pthfile, torch.device('cpu'))  # 设置在cpu环境下查询
# print(pthfile.split("/")[-1].split(".")[0])
with open(pthfile.split("/")[-1].split(".")[0]+".txt","w+") as f:
    for k in model:  # 查看模型字典里面的key
        f.write(str(model[k].shape)+"\n")

# for k in model:  # 查看模型字典里面的value
#     print(model[k].shape)

# for k in model.keys():  # 查看模型字典里面的key
#     print(k)

# print('type:')
# print(type(model))  # 查看模型字典长度
# print('length:')
# print(len(model))
# print('key:')
# for k in model.keys():  # 查看模型字典里面的key
#     print(k)
# print('value:')
# for k in model:  # 查看模型字典里面的value
#     print(k, model[k])