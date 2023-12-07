### 介绍

nn.module是pytorch框架中搭建神经网络的基础，所有的神经网络都需要继承这个模块。

ps:这里还是仔细看官方文档

```python
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self): 
        super().__init__() 

    def forward(self, input):  # 前向传播
        output = input + 23
        return output

model = Model()  
x = torch.tensor(1.0)  
output = model(x)  
print(output)

```

这块就是讲了神经网络继承下这个类，找个简单神经网络写一下就懂了。
