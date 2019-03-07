# Short-message-classification
Short message Multi-classification with LGBM



用LGBM为中文垃圾短信分类，为了提升模型在数据流中的运行速度，将特征中要使用到的`countVec`在训练时用`pickle`打包，且在预处理和预测输出循环中使用了`multiprocessing`模块。在`Pred.py`中，将`cores`值设置成<=你的CPU核心数以提升预测和预处理的速度。 
<br></br>

## File
* pred.py  
multiprocessing预测文件

* lgbm_train.py  
训练文件

* process.py  
预处理

* train/test  
训练集和测试集
