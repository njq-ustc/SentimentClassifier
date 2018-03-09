# -*- coding: utf-8 -*-
# 文本预处理，清洗数据
import jieba
import re
from langconv import *

# 创建停用词list  
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 对句子进行分词  
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist(r'..\test\sentiment-analysis\chinese_stop_words.txt')  #加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != "\t":
                outstr += word
    return outstr


input = open(r'..\test\sentiment-analysis\positive.txt', 'r', encoding='utf-8')
output = open(r'..\test\sentiment-analysis\positive_process.txt', 'w')

for line in input:
    line_seg = seg_sentence(line).strip()  # 这里的返回值是字符串
    p = re.compile(r'[^\u4e00-\u9fa5]')  # 只匹配中文

    line_seg = " ".join(p.split(line_seg)).strip()
    # line_seg = re.sub(r"[\U00010000-\U0010ffff]","",line_seg)#去除表情
    line_seg = re.sub(r"\n", "", line_seg)
    line_seg = re.sub(r"\r", "", line_seg)
    # line_seg = re.sub(r"[a-zA-Z0-9]*", "", line_seg)  # 去除数字和英文
    line_seg = re.sub(r"\s{2,}", " ", line_seg)  # 将多个空格转化为一个空格
    line_seg = re.sub(r" ", "", line_seg)
    line_seg = line_seg.strip()
    if (line_seg == ' ') | (line_seg ==''):#去除空行
         continue

    #转换繁体到简体
    line_seg = Converter('zh-hans').convert(line_seg)

    # 转换简体到繁体
    # line = Converter('zh-hant').convert(line.decode('utf-8'))

    output.write(line_seg + '\n')
output.close()
input.close()
