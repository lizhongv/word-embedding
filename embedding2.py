import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

SAMPLE_SIZE = 3
EMBEDDING_DIM = 10
yuliao = """晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷，
渔人甚异之。复前行，欲穷其林。林尽水源，便得一山，山有小口，仿佛若有光。便舍船，从口入。初极狭，才通人。复行数十步，豁然开朗。
土地平旷，屋舍俨然，有良田美池桑竹之属。阡陌交通，鸡犬相闻。其中往来种作，男女衣着，悉如外人。黄发垂髫，并怡然自乐。
见渔人，乃大惊，问所从来。具答之。便要还家，设酒杀鸡作食。村中闻有此人，咸来问讯。自云先世避秦时乱，率妻子邑人来此绝境，不复出焉，遂与外人间隔。
问今是何世，乃不知有汉，无论魏晋。此人一一为具言所闻，皆叹惋。余人各复延至其家，皆出酒食。停数日，辞去。此中人语云：“不足为外人道也。
”既出，得其船，便扶向路，处处志之。及郡下，诣太守，说如此。太守即遣人随其往，寻向所志，遂迷，不复得路。南阳刘子骥，高尚士也，闻之，欣然规往。
未果，寻病终，后遂无问津者。""".strip()

train_dataset = [ ([yuliao[i], yuliao[i + 1], yuliao[i + 2]], yuliao[i + 3])
                 for i in range(len(yuliao) - 3)]

vocab = set(yuliao)  # 语料库
word_dict = {word: key for key, word in enumerate(vocab) }

# print(word_dict, len(word_dict))

class NGramCivilNet(nn.Module):  # 以3-gram 方式训练 word embedding
    def __init__(self, vocab_size, embedding_dim, sample_size):
        super(NGramCivilNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(sample_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1)) # [1, sample_size * embedding_dim]
        print(embeds.shape)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

loss_function = nn.NLLLoss()  # torch.nn.CrossEntropyLoss 相当于 softmax + log + nllloss。
model = NGramCivilNet(len(vocab), EMBEDDING_DIM, SAMPLE_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

print(model.parameters)
# for x in model.parameters():
#     print(f"this layer have {x.numel()} paramerters")
# print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
# print("The model have {} paramerters needed to calculate the gradient in total".format(sum(x.numel() for x in model.parameters() if x.requires_grad)))
# parameters = filter(lambda x: x.requires_grad, model.parameters())
# print(sum(x.numel() for x in parameters))
# model.state_dict()

for epoch in range(100):
    total_loss = 0

    for sample, target in train_dataset:
        sample_idxs = torch.tensor([word_dict[w] for w in sample], 
                                    dtype=torch.long)

        model.zero_grad()  # 梯度归零，否则累加
        log_probs = model(sample_idxs)
        # NLLoss()使用方法
        loss = loss_function(log_probs, torch.tensor([word_dict[target]],
                                                     dtype=torch.long))  
        loss.backward()   # 反向传播逐层计算梯度
        optimizer.step()  # 更新参数
    print("epoch:{}  gemfield loss: {}".format(epoch, loss.item()))  # loss.item()变为标量
    # print(model.embeddings.weight[word_dict["山"]])

# To get the embedding of a particular word, e.g. "山"
# print(model.embeddings.weight)
# print(model.embeddings.weight.size)
# print(model.embeddings.weight[word_dict["山"]])
