from TrainingSet import load_data  # 加载数据函数
from NeuralNetworkArchitecture import NeuralNetwork  # 导入神经网络类

# Step 2: 加载训练数据
X_train, y_train = load_data()
input_dim = X_train.shape[1]  # 特征数量
hidden_dim = 10               # 隐藏层大小
output_dim = 1                # 输出层大小（适用于二分类）

# Step 3: 初始化神经网络模型
model = NeuralNetwork(input_dim, hidden_dim, output_dim)

# Step 4: 定义训练参数
epochs = 1000          # 训练的轮数
learning_rate = 0.01   # 学习率

# Step 5: 训练模型
for epoch in range(epochs):
    output = model.forward(X_train)  # 前向传播
    loss = model.compute_loss(output, y_train)  # 计算损失
    model.backward(X_train, y_train, learning_rate)  # 反向传播并更新权重
    
    # 每 100 个 epoch 输出一次损失
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

print("Training complete.")
