import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # 这里的参数，SGD里面的，只要定义两个参数，一个是tudui.parameters()本身，另一个是lr

# 设置训练网络的一些参数

# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("------------第 {} 轮训练开始------------".format(i + 1))

    # 训练步骤开始
    tudui.train()  # 这两个层，只对一部分层起作用，比如 dropout层；如果有这些特殊的层，才需要调用这个语句
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 优化器，梯度清零
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))  # 这里用到的 item()方法，有说法的，其实加不加都行，就是输出的形式不一样而已
            writer.add_scalar("train_loss", loss.item(), total_train_step)  # 这里是不是在画曲线？

    # 每训练完一轮，进行测试，在测试集上测试，以测试集的损失或者正确率，来评估有没有训练好，测试时，就不要调优了，就是以当前的模型，进行测试，所以不用再使用梯度（with no_grad 那句）

    # 测试步骤开始
    tudui.eval()  # 这两个层，只对一部分层起作用，比如 dropout层；如果有这些特殊的层，才需要调用这个语句
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 这样后面就没有梯度了，  测试的过程中，不需要更新参数，所以不需要梯度？
        for data in test_dataloader:  # 在测试集中，选取数据
            imgs, targets = data
            outputs = tudui(imgs)  # 分类的问题，是可以这样的，用一个output进行绘制
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()  # 为了查看总体数据上的 loss，创建的 total_test_loss，初始值是0
            accuracy = (outputs.argmax(1) == targets).sum()  # 正确率，这是分类问题中，特有的一种，评价指标，语义分割之类的，不一定非要有这个东西，这里是存疑的，再看。
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))  # 即便是输出了上一行的 loss，也不能很好的表现出效果。
    # 在分类问题上比较特有，通常使用正确率来表示优劣。因为其他问题，可以可视化地显示在tensorbo中。
    # 这里在（二）中，讲了很复杂的，没仔细听。这里很有说法，argmax（）相关的，有截图在word笔记中。
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))  # 保存方式一，其实后缀都可以自己取，习惯用 .pth。
    print("模型已保存")

writer.close()

