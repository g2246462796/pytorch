torchvision 是pytorch框架专门处理数据的一个包。

这里up主要讲的是从torchvision下载比较常用的数据集。

```python
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# 一般对数据可能要做很多处理，我们直接初始化一个compose，把需要的处理写一起
# 数据集是PIL，需要转Tensor,这里照片很小，我们不做其他操作

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

print(test_set[0])
print(test_set.classes)

img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])

print(test_set[0])
writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
```

参数说明:

​	root  存储路径

​    train  True/False  训练集or测试集

​	transform 进行变换

​	download   True/False  下载or不下载。建议写True。

后面常规操作了，相当于复习下前面的内容。