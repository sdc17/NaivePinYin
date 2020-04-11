# 拼音输入法

### 实验环境
> Ubuntu 18.04


### 安装依赖
涉及到的包常用的conda/pip源应该都能下载到。


### 文件内容
> predict.py 用于实际测试
>
> preprocess.py 生成gram和word统计文件
>
> eval.py 用于测群里的数据集
>
> script.py 测试脚本

### 模型使用
性能最好的是```全三元字模型```
##### 二元字模型
```bash
python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=2c
```

##### 20%三元字模型

```bash
python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=3c --full_model=False
```

##### 全三元字模型
```bash
python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=3c --full_model=True
```

##### 20%二元词模型
```bash
python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=2w --full_model=False
```

##### 全二元词模型
```bash
python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=2w --full_model=True
```

也可以拿```script.py```来测，里面有各个模型用法相应的注释。