from torch.utils.tensorboard import SummaryWriter
# 一个类，往事件文件夹里写东西

writer = SummaryWriter("logs")

for i in range(100):
  writer.add_scalar("y=3x",3*i,i)

writer.close()
