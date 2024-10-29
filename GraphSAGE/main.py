import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 加载数据
# 假设数据中第一列是数据序号，最后一列是标签，中间的列是特征
df = pd.read_csv(r"dataset\Fe2+-data.csv", header=0)
# 检查每列的缺失值数量
#print(df.isnull().sum())
X = df.iloc[:, 2:99].values # 选择除了第一列和第2列之外的所有列作为特征
y = df.iloc[:,1].values # 选择第2列作为标签

# 标准化特征,使用StandardScaler对特征矩阵X进行标准化处理，使得特征均值为0，方差为1。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 为每个数据子集重新计算 KNN 图的邻接矩阵
k = 3
train_A = kneighbors_graph(X_train, k, mode='connectivity', include_self=True).tocoo()
test_A = kneighbors_graph(X_test, k, mode='connectivity', include_self=True).tocoo()

# 创建边索引,用于构建图数据对象。
train_edge_index = torch.tensor(np.vstack([train_A.row, train_A.col]), dtype=torch.long)
test_edge_index = torch.tensor(np.vstack([test_A.row, test_A.col]), dtype=torch.long)

# 创建 PyTorch Geometric 数据对象,包括特征张量x、边索引edge_index和标签y。
train_data = Data(x=torch.tensor(X_train, dtype=torch.float), edge_index=train_edge_index, y=torch.tensor(y_train, dtype=torch.long))
test_data = Data(x=torch.tensor(X_test, dtype=torch.float), edge_index=test_edge_index, y=torch.tensor(y_test, dtype=torch.long))

# 定义 GraphSAGE 模型,是一种用于图数据的深度学习模型，旨在学习节点的表示
#GraphSAGE的工作流程可以总结如下：
#邻居采样：对于每个节点，从其邻居中采样一定数量的节点。
#特征聚合：将采样到的邻居节点的特征进行聚合，生成新的节点表示。
#多层采样和聚合：重复多次邻居采样和特征聚合过程，每一层可以使用不同的采样方式和聚合函数。
#学习节点表示：通过监督学习或无监督学习方法，学习每个节点的表示，优化节点表示与节点标签（如果有的话）之间的匹配或相似性。

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 初始化和训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGENet(train_data.num_node_features, 16, len(np.unique(y))).to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
criterion = torch.nn.NLLLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    loss = criterion(out, train_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# 最终测试和结果存储函数
def final_test_and_store_results():
    model.eval()
    out_test = model(test_data)
    pred_test = out_test.argmax(dim=1)
    overall_test_acc = (pred_test == test_data.y).sum().item() / len(test_data.y)

    with open(r"dataset\test_results.txt", 'w') as f:
        f.write("True\tPredicted\n")
        for t, p in zip(test_data.y.cpu().numpy(), pred_test.cpu().numpy()):
            f.write(f"{t}\t{p}\n")
        f.write("\n")
    print(f'Overall Test Accuracy: {overall_test_acc:.4f}')

# 新增一个方法来绘制训练精度和损失值,制训练精度和损失值图在一张图上
def plot_metrics(train_accuracies, train_losses):
    plt.figure(figsize=(10, 6))

    # 绘制训练精度
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='blue')
    #plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy',color='blue')
    plt.tick_params(axis='y', labelcolor='blue')
    plt.legend(loc='upper left')

    # 创建第二个y轴并绘制训练损失
    plt.twinx()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='red')
    #plt.plot(train_losses, label='Train Loss', color='red')
    plt.ylabel('Loss',color='red')
    plt.tick_params(axis='y', labelcolor='red')
    plt.legend(loc='upper right')
    # 设置 x 轴刻度为整数，并以 5 的间距显示
    plt.xticks(np.arange(0, len(train_accuracies) + 1, 25))

    plt.title('Training Accuracy and Loss')
    plt.tight_layout()
    # 保存图像到文件
    plt.savefig(r'dataset\Training Accuracy and Loss.png')
    plt.show()


# 训练和测试模型
train_losses = []
train_accuracies = []
# 打开文件以追加写入模式
result_file = open(r"dataset\training_results.txt", 'a')
for epoch in range(1, 401):  # 训练次数
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    loss = criterion(out, train_data.y)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    # 计算训练集的训练精度
    model.eval()
    out_train = model(train_data)
    pred_train = out_train.argmax(dim=1)
    correct_train = (pred_train == train_data.y).sum().item()
    train_acc = correct_train / len(train_data.y)
    train_accuracies.append(train_acc)
    # 每次训练后输出和存储测试结果
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}')
    # 将结果写入文件
    result_file.write(f'Epoch {epoch}: Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}\n')


# 进行最终的测试和结果存储
final_test_and_store_results()
# 绘制训练精度和损失值
plot_metrics(train_accuracies, train_losses)