# Short-message-classification
Short message Multi-classification with LGBM



用LGBM为中文垃圾短信分类，为了提升模型在数据流中的运行速度，将特征中要使用到的`countVec`在训练时用`pickle`打包，且在预处理和预测输出循环中使用了`multiprocessing`模块。在`Pred.py`中，将`cores`值设置成<=你的CPU核心数以提升预测和预处理的速度。 
<br></br>

## File
* pred.py  
`multiprocessing`预测文件，其中的测试集需要你从原本含有标签的训练集中自己划分，然后将文件名传入`excel_to_list`中。输出csv文件名可以自己定义。

* lgbm_train.py  
训练文件，为了提升预测速度，我们会把`model`和`countVec`加载到本地以便`pred.py`随时读取。`Score`函数是用于简单计算预测正确标签的个数，第二个参数需要传入测试集的正确名称。

* process.py  
预处理

* train  
训练集
