# Tweet Sentiment Extraction

[TOC]

## 1、EDA

### 1.统计各类情感数量

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-11-225726.png" alt="image-20230311225725141" style="zoom: 67%;" />

### 2.统计各文本长度

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-11-225740.png" alt="image-20230311225738974" style="zoom:67%;" />

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-11-225749.png" alt="image-20230311225748230" style="zoom:67%;" />

## 2、模型训练

### 1.Bert

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-172514.png" alt="image-20230328172512776" style="zoom:67%;" />

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-172454.png" alt="image-20230328172453553" style="zoom:67%;" />

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-172528.png" alt="image-20230328172527233" style="zoom:67%;" />

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-172506.png" alt="image-20230328172504742" style="zoom:67%;" />

```python
test_loss:0.524, test_jaccards:0.5036
```

### 2.T5

![image-20230328173020179](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-173021.png)

![image-20230328173219345](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-173220.png)

![image-20230328173229093](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-173230.png)

```pyhton
test_loss:0.223, test_jaccards:0.6549
```

### 3.GPT2

![image-20230328173404282](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-173405.png)

![image-20230328173512896](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-173514.png)

![image-20230328173504290](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-28-173505.png)

```python
test_loss:0.8083, test_jaccards:0.6351
```

## 4、T5_3B_lora

![image-20230409235041657](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-04-09-235044.png)

![image-20230409235110795](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-04-09-235112.png)

![image-20230409235125116](https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-04-09-235126.png)

```python
'test_loss': 0.2015, 'test_jaccard': 0.7087
```

