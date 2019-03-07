import xlrd
#import jieba_fast as jieba
#import jieba
import re
import pandas as pd
import jieba_fast as jieba

jieba.initialize()




#0正常 1贷款 2信用卡 3广告 4其他
def preProcess(flag = 'train'):
    

    with open(flag + '.csv', 'w', encoding='utf-8') as w:
        if flag == 'test':
            fix = ''
            end = '\n'
            col = 0
        else:
            fix = ',label'
            end = ''
            col = 0

        w.write('text' + fix +'\n')
        workbook = xlrd.open_workbook(flag + '.xlsx')
        for x in range(0, len(workbook.sheets())):
            sheet = workbook.sheet_by_index(x)
            for row in range(1, sheet.nrows):
                value = sheet.cell(row,col).value
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

                if flag == 'train':
                    if x == 0:
                        outer += ',0\n'
                    elif x == 1:
                        outer +=',1\n'
                    elif x == 2:
                        outer +=',2\n'
                    elif x == 3:
                        outer += ',3\n'
                    else:
                        outer += ',4\n'
                w.write(outer + end)
    



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






def excel_to_list(test):
    workbook = xlrd.open_workbook(test + '.xlsx')
    sheet = workbook.sheet_by_index(0)
    rows = sheet.nrows
    a = []
    for row in range(0, rows):
        a.append(sheet.cell(row, 0).value)
    return a






def read_excel(fp, x):
    
    workbook = xlrd.open_workbook(fp + '.xlsx')
    sheet = workbook.sheet_by_index(0)
    rows = sheet.nrows
    con = 625 
    a = []
    for row in range(1 + x * con, con + 1 + x * con):
        a.append(sheet.cell(row, 0).value)
    
    return a



