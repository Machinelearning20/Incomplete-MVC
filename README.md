## 运行
1. 检查`configs`中是否有这个数据集对应的`.yaml`文件，如果没有，可以复制已有数据集的`.yaml`，修改`dataset`属性和对应的超参。
2. 使用命令`python -u main.py -d $dataset`运行指定的数据集。
3. 模型默认保存在`results/$dataset`文件夹中
4. `main.py`中的device来指定使用设备，默认是第0张显卡