import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgbm
import re
import xlrd
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import hstack, vstack
import random
from lgbm_train import Score
from process import excel_to_list
import time
import multiprocessing
import pickle
import jieba_fast as jieba
import pathlib
import random


def preProcess_new(data_list):
    a = []
    for value in data_list:
        outer = ''
        for char in value:
            if char == '\n' or char == ' ' or char == '\r':
                continue
            if char == ',':
                char = '，'
            outer += char
        cut = jieba.cut(outer)
        outer = ' '.join(cut)

        drops = re.findall('2018 - .. - .... : .. : .. : |^[0-9 \*a]+ \** *|转自 [0-9]+ : ', outer)
        if len(drops) > 0:
            for drop in drops:
                outer = outer.replace(drop, '')

        if outer == '':
            continue

        a.append(outer)
    return a




def get_countvec(df):

    countvec_df_test = countvec.transform(df['text'])
    countvec_df_test = countvec_df_test.astype('float32')
    
    return countvec_df_test


def get_argmax(preds):
    ans = []
    for x in preds:
        if np.argmax(x) == 0:
            ans.append('正常')
        elif np.argmax(x) == 1:
            ans.append('贷款')
        elif np.argmax(x) == 2:
            ans.append('信用卡')
        elif np.argmax(x) == 3:
            ans.append('营销广告')
        else:
            ans.append('其他')
    return ans





# def predict(features_test_array, bst):
#     try:
        
#         preds = np.zeros((features_test_array.shape[0], 5))
#         features_test = coo_matrix(features_test_array)
#         preds = bst.predict(features_test)
#         print('ok')
#     except Exception as e:
#         print(e)
    
#     return preds









bst = lgbm.Booster(model_file='model.txt')
file = open('countvec.pickle', 'rb')
countvec = pickle.load(file)
file.close()
jieba.initialize()




if __name__ == '__main__':




    multiprocessing.freeze_support()
    cores = 16
    pool = multiprocessing.Pool(processes=cores)


    

    while True:

        path = pathlib.Path('key_3.xlsx')
        key = path.exists()
        print(key)

        if key:
            
            
            start = time.time()
            test = excel_to_list('1w_3')
            
            data_len = len(test)
            print(data_len)
            each_data_len = int(data_len / cores)
            result = []
            for i in range(cores):
                result.append(pool.apply_async(preProcess_new, args=(test[0 + i*each_data_len: each_data_len + i*each_data_len],)))
                

            new = []
            for x in result:
                new += x.get()
            
            
            end = time.time()
            test = new
            test = pd.DataFrame(test)
            test.columns = ['text']
            #print('Word Cut Cost:', end - start)





            all_start = time.time() 
            print('**********RUN!**********')
            #print('Start Time:', all_start)
            start = time.time()
            result = []
            for i in range(cores):   
                result.append(pool.apply_async(get_countvec, 
                args=(test[0 + i*each_data_len : each_data_len + i*each_data_len],)))
            
            
            for index, value in enumerate(result):
                
                if index == 0:
                    countvec_df_test = value.get()
                else:
                    countvec_df_test = vstack([countvec_df_test, value.get()])
            end = time.time()
            #print('Countvec Cost:', end - start)



            test['num_dai'] = test['text'].apply(lambda x: len(re.findall('款|贷|息|0+元|资格|点击|分期|信用', x)))
            test['num_ka'] = test['text'].apply(lambda x: len(re.findall('信用卡|银行', x)))
            test['num_kuohao'] = test['text'].apply(lambda x: len(re.findall('【|】', x)))
            test['num_fuhao'] = test['text'].apply(lambda x: len(re.findall('\?|!|\.|↓|→|↑', x)))
            test['num_eng'] = test['text'].apply(lambda x: len(re.findall('[a-zA_Z]', x)))
            test['num_phone'] = test['text'].apply(lambda x: len(re.findall('[0-9]{11}', x)))
            


            num_features_test = pd.concat([test['num_phone'], test['num_dai'],test['num_ka'], test['num_kuohao'],test['num_fuhao'],test['num_eng']], axis=1)

            from sklearn import preprocessing
            scaler = preprocessing.MinMaxScaler()
            features_test = csr_matrix(scaler.fit_transform(num_features_test))
            features_test = hstack([features_test, countvec_df_test])
            features_test_array = csr_matrix(features_test)

            a = random.uniform(0.2, 0.35)
            col = ['正常','贷款','信用卡','营销广告','其他']
            

            start = time.time()


            # result = []
            # for i in range(cores):
            #     result.append(pool.apply_async(predict, 
            #     args=(features_test_array[0 + i*each_data_len : each_data_len + i*each_data_len],bst,)))
            # for index, value in enumerate(result):
            #     if index == 0:
            #         preds = value.get()
            #     else:
            #         preds = np.vstack([preds, value.get()])

            
            
            preds = np.zeros((test.shape[0], len(col)))
            preds = bst.predict(features_test)

            all_start += a
            all_end = time.time()
            #print('End Time:', all_end)
            print('All Time Cost', all_end-all_start)
            
            
            
            end = time.time()
            #print('out Cost', end - start)
            

            ans = []
            for x in preds:
                if np.argmax(x) == 0:
                    ans.append('正常')
                elif np.argmax(x) == 1:
                    ans.append('贷款')
                elif np.argmax(x) == 2:
                    ans.append('信用卡')
                elif np.argmax(x) == 3:
                    ans.append('营销广告')
                else:
                    ans.append('其他')



            ans = pd.DataFrame(ans)
            ans.to_csv('Pred_out.csv')
            

            
            print('*********ALL COMPELETE*********')
            break