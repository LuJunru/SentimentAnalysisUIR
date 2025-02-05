# 介绍
- 本项目是针对港中文MODEST系统中情感分析子系统（简称M系统）的重构

# 历史
- M系统的设计原理是：
  - 训练时：基于情感词典从句子中抽取出情感词、否定词和修饰词作为特征，将特征放入三分类SVM中训练（positive、neural & negative）
  - 使用时：将文章打散为5类计算权重不同的句子，计算每个句子的情感分数，加权求和得到文章的情感得分。若得分落在[-1, lower阈值]则为negative；
    若得分落在[upper阈值，1]则为positive；否则为neural
- M系统的精确度是：F1为74%

# 改进
- 直接使用词向量和LSTM模型提取句子特征并进行分类，然后沿用句子分类及权重标注和文章情感分计算方法
- 在同样的数据集上进行精确度测试

# 备注
- 程序中用到的词向量下载地址: [链接](https://pan.baidu.com/s/1q-nldm4HQe0P5dL8Zu4QPw)，提取码是kpue
