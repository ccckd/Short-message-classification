import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgbm
import re
import xlrd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import random
from process import preProcess




def Score(csv, name):
    score = 0
    pred_df = pd.read_csv(csv)   
    workbook = xlrd.open_workbook(name + '.xlsx')
    for x in range(0, 1):
        sheet = workbook.sheet_by_index(x) # sheet索引从0开始
        for row in range(1, sheet.nrows):
            try:
                std = sheet.cell(row,1).value
                pred = pred_df.iloc[row - 1, 1]
            except:
                pred = 'None'   
            if pred == std:
                score += 1
    print(score)
    return score      



 


if __name__ == "__main__":

    preProcess(flag='train')
    preProcess(flag='test')


    # loading data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')


    train_ = train['text']
    test_ = test['text']
    label = train['label']


    alldata = pd.concat([train_, test_], axis=0)
    alldata = pd.DataFrame(alldata)




    tfidf = TfidfVectorizer(
                max_df = 0.85,
                max_features = 5000,
                ngram_range = (1, 6),
                use_idf = 1,
                smooth_idf = 1,
                )
    tfidf = tfidf.fit(alldata['text'])
    tfidf = tfidf.fit_transform(alldata['text'])
    tfidf_df_train = tfidf[:len(train_)]
    tfidf_df_test = tfidf[len(train_):]
    tfidf_df_train = tfidf_df_train.astype('float32')
    tfidf_df_test = tfidf_df_test.astype('float32')



    import pickle
    countvec = CountVectorizer(max_features = 5000, ngram_range=(1, 7), token_pattern=r"(?u)\b\w+\b",)
    countvecdata = countvec.fit_transform(alldata['text'])
    file = open('countvec.pickle', 'wb')
    pickle.dump(countvec, file)
    file.close()
    countvec_df = pd.DataFrame(countvecdata.todense()) 
    countvec_df.columns = ['col' + str(x) for x in countvec_df.columns]
    countvec_df_train = countvecdata[:len(train_)] 
    countvec_df_test = countvecdata[len(train_):]
    countvec_df_train_ = countvec_df_train.astype('float32')
    countvec_df_test_ = countvec_df_test.astype('float32')







    train['num_dai'] = train['text'].apply(lambda x: len(re.findall('款|贷|息|0+元|资格|点击|分期|信用', x)))
    test['num_dai'] = test['text'].apply(lambda x: len(re.findall('款|贷|息|0+元|资格|点击|分期|信用', x)))


    train['num_ka'] = train['text'].apply(lambda x: len(re.findall('信用卡|银行', x)))
    test['num_ka'] = test['text'].apply(lambda x: len(re.findall('信用卡|银行', x)))


    train['num_kuohao'] = train['text'].apply(lambda x: len(re.findall('【|】', x)))
    test['num_kuohao'] = test['text'].apply(lambda x: len(re.findall('【|】', x)))

    train['num_fuhao'] = train['text'].apply(lambda x: len(re.findall('\?|!|\.|↓|→|↑', x)))
    test['num_fuhao'] = test['text'].apply(lambda x: len(re.findall('\?|!|\.|↓|→|↑', x)))

    train['num_eng'] = train['text'].apply(lambda x: len(re.findall('[a-zA_Z]', x)))
    test['num_eng'] = test['text'].apply(lambda x: len(re.findall('[a-zA_Z]', x)))

    train['num_phone'] = train['text'].apply(lambda x: len(re.findall('[0-9]{11}', x)))
    test['num_phone'] = test['text'].apply(lambda x: len(re.findall('[0-9]{11}', x)))

    train['len'] = train['text'].apply(lambda x: len(x))
    test['len'] = test['text'].apply(lambda x: len(x))


    num_features_train = pd.concat([train['num_phone'], train['num_dai'],train['num_ka'], train['num_kuohao'], train['num_fuhao'],train['num_eng']],axis=1)
    num_features_test = pd.concat([test['num_phone'], test['num_dai'],test['num_ka'], test['num_kuohao'],test['num_fuhao'],test['num_eng']], axis=1)



    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    features = csr_matrix(scaler.fit_transform(num_features_train))
    features_test = csr_matrix(scaler.fit_transform(num_features_test))



    features = hstack([features, countvec_df_train_])
    features_test = hstack([features_test, countvec_df_test_])





    col = ['正常','贷款','信用卡','营销广告','其他']
    preds = np.zeros((test.shape[0], len(col)))

    params = {
        'objective' :'multiclass',
        'num_class': 5,
        'learning_rate' : 0.009, #0.008
        'num_leaves' : 74,
        'feature_fraction': 0.84, 
        'bagging_fraction': 0.35, 
        'bagging_freq':1,
        'boosting_type' : 'gbdt',
        'metric': 'multi_logloss'
    }

    from sklearn.model_selection import train_test_split

    X_train, X_valid, Y_train, Y_valid = train_test_split(features, label, random_state=7, test_size=0.25)
    X_best = SelectKBest(f_classif, k=20).fit_transform(X_train, Y_train)

    d_train = lgbm.Dataset(X_train, Y_train)
    d_valid = lgbm.Dataset(X_valid, Y_valid)

    # training with early stop
    bst = lgbm.train(params, d_train, 2500, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=200)
    bst.save_model('model.txt')
    # bst = lgb.Booster(model_file='model.txt')


    # making prediciton for one column
    import time

    time_start = time.time()
    preds = bst.predict(features_test)
    time_end = time.time()
    print('time cost:', time_end - time_start)




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
    ans.to_csv('lgbm_out.csv')


    Score('lgbm_out.csv', '1w')











