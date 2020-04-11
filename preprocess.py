#-*- coding : utf-8-*-

import os
import json
import yaml
import glob
import jieba
import pickle
import datetime
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, Executor, as_completed


def task_one_gram(news):

    gram1 = {}
    with open('./training/pinyin_table/一二级汉字表.txt', encoding='gbk') as f:
        keys = list(f.read().replace('\n', ''))
        for key in keys:
            gram1[key] = 0

    for line in open(news, 'r', encoding='gbk', errors='ignore'):
        try:
            content = json.loads(line)
        except:
            continue
        info = content['html'] + content['title']
        word_dict = Counter(info)
        for key, value in word_dict.items():
            if gram1.get(key) != None:
                gram1[key] += value
    return gram1


def one_gram():
    news = glob.glob('./training/sina_news_gbk/*.txt')
    gram1 = {}

    with ProcessPoolExecutor() as executor:
        for results in executor.map(task_one_gram, news):
            for key, value in results.items():
                gram1[key] = gram1.get(key, 0) + value

    with open('./data/1gram.pkl', 'wb') as f:
        pickle.dump(gram1, f)

    # with open('./data/1gram.yaml', 'w', encoding='utf-8') as f:
    #     yaml.dump(gram1, f, default_flow_style=False, allow_unicode=True)
            

def task_two_gram(news):

    gram1 = {}
    gram2 = {}
    cnt_s = 0
    cnt_t = 0
    with open('./data/1gram.pkl','rb') as f:
        gram1 = pickle.load(f)

    for line in open(news, 'r', encoding='gbk', errors='ignore'):
        try:
            content = json.loads(line)
        except:
            continue
        info = content['html'] + content['title']
        info_len = len(info)
        for i in range(info_len - 1):
            key = None
            if i == 0 and gram1.get(info[0]):
                key = ''.join(['s', info[0]])
                cnt_s += 1
            elif i > 0 and gram1.get(info[i]) and not gram1.get(info[i - 1]):
                key = ''.join(['s', info[i]])
                cnt_s += 1
            elif gram1.get(info[i]) and not gram1.get(info[i + 1]):
                key = ''.join([info[i], 't'])
                cnt_t += 1
            elif i == info_len - 2 and gram1.get(info[i + 1]):
                key = ''.join([info[i + 1], 't'])
                cnt_t += 1
            if key:
                gram2[key] = gram2.get(key, 0) + 1

            if gram1.get(info[i]) and gram1.get(info[i + 1]):
                key = ''.join([info[i], info[i + 1]])
                gram2[key] = gram2.get(key, 0) + 1
            
    return (gram2, cnt_s, cnt_t)
    

def two_gram():
    
    news = glob.glob('./training/sina_news_gbk/*.txt')
    gram1 = {}
    gram2 = {}
    cnt_s = 0
    cnt_t = 0
    with ProcessPoolExecutor() as executor:
        for results in executor.map(task_two_gram, news):
            for key, value in results[0].items():
                gram2[key] = gram2.get(key, 0) + value
                cnt_s += results[1]
                cnt_t += results[2]

    with open('./data/1gram.pkl','rb') as f:
        gram1 = pickle.load(f)
    gram1['s'] = cnt_s
    gram1['t'] = cnt_t
    with open('./data/1gram.pkl', 'wb') as f:
        pickle.dump(gram1, f)
    # with open('./data/1gram.yaml', 'w', encoding='utf-8') as f:
    #     yaml.dump(gram1, f, default_flow_style=False, allow_unicode=True)
        
    with open('./data/2gram.pkl', 'wb') as f:
        pickle.dump(gram2, f)

    # with open('./data/2gram.yaml', 'w', encoding='utf-8') as f:
    #     yaml.dump(gram2, f, default_flow_style=False, allow_unicode=True)


def task_three_gram(news):
    gram1 = {}
    gram3 = {}
    cnt_s = 0
    cnt_t = 0
    with open('./data/1gram.pkl','rb') as f:
        gram1 = pickle.load(f)

    for line in open(news, 'r', encoding='gbk', errors='ignore'):
        try:
            content = json.loads(line)
        except:
            continue
        info = content['html'] + content['title']
        info_len = len(info)
        for i in range(info_len - 2):
            key = ''
            if i == 0 and gram1.get(info[0]) and gram1.get(info[1]):
                key = ''.join(['s', info[0], info[1]])
                cnt_s += 1
            elif i > 1 and gram1.get(info[i]) and gram1.get(info[i - 1]) and not gram1.get(info[i - 2]):
                key = ''.join(['s', info[i - 1], info[i]])
                cnt_s += 1
            elif gram1.get(info[i]) and gram1.get(info[i + 1]) and not gram1.get(info[i + 2]):
                key = ''.join([info[i], info[i + 1], 't'])
                cnt_t += 1
            elif i == info_len - 3 and gram1.get(info[i + 1]) and gram1.get(info[i + 2]):
                key = ''.join([info[i + 1], info[i + 2], 't'])
                cnt_t += 1
            if key:
                gram3[key] = gram3.get(key, 0) + 1

            if gram1.get(info[i]) and gram1.get(info[i + 1]) and gram1.get(info[i + 2]):
                key = ''.join([info[i], info[i + 1], info[i + 2]])
                gram3[key] = gram3.get(key, 0) + 1
            
    return (gram3, cnt_s, cnt_t)


def three_gram(rank):
    news = glob.glob('./training/sina_news_gbk/*.txt')
    gram1 = {}
    gram3 = {}
    cnt_s = 0
    cnt_t = 0
    with ProcessPoolExecutor() as executor:
        for results in executor.map(task_three_gram, news):
            for key, value in results[0].items():
                gram3[key] = gram3.get(key, 0) + value
                cnt_s += results[1]
                cnt_t += results[2]

    with open('./data/1gram.pkl','rb') as f:
        gram1 = pickle.load(f)
    gram1['s'] = cnt_s
    gram1['t'] = cnt_t
    with open('./data/1gram.pkl', 'wb') as f:
        pickle.dump(gram1, f)
    # with open('./data/1gram.yaml', 'w', encoding='utf-8') as f:
    #     yaml.dump(gram1, f, default_flow_style=False, allow_unicode=True)
    
    # Select top rank% features for valid features
    gram3 = dict(sorted(gram3.items(), key = lambda x: x[1], reverse = True)[:int(len(gram3)*rank)])

    if rank == 1.0:
        with open('./data/3gram_whole.pkl', 'wb') as f:
            pickle.dump(gram3, f)

        # with open('./data/3gram_whole.yaml', 'w', encoding='utf-8') as f:
        #     yaml.dump(gram3, f, default_flow_style=False, allow_unicode=True)
    else:
        with open('./data/3gram.pkl', 'wb') as f:
            pickle.dump(gram3, f)

        # with open('./data/3gram.yaml', 'w', encoding='utf-8') as f:
        #     yaml.dump(gram3, f, default_flow_style=False, allow_unicode=True)


def task_one_word(news):

    word1 = {}
    gram1 = {}

    with open('./data/1gram.pkl','rb') as f:
        gram1 = pickle.load(f)

    for line in open(news, 'r', encoding='gbk', errors='ignore'):
        try:
            content = json.loads(line)
        except:
            continue
        info = content['html'] + content['title']
        seg = jieba.lcut(info, cut_all=False, HMM=True)
        word_dict = Counter(seg)
        for key, value in word_dict.items():
            valid = True
            for i in key:
                if not gram1.get(i) or i == 's' or i == 't':
                    valid = False
                    break
            if valid:
                word1[key] = word1.get(key, 0) + value
    return word1


def one_word():
    news = glob.glob('./training/sina_news_gbk/*.txt')
    word1 = {}

    with ProcessPoolExecutor() as executor:
        for results in executor.map(task_one_word, news):
            for key, value in results.items():
                word1[key] = word1.get(key, 0) + value

    with open('./data/1word.pkl', 'wb') as f:
        pickle.dump(word1, f)

    # with open('./data/1word.yaml', 'w', encoding='utf-8') as f:
    #     yaml.dump(word1, f, default_flow_style=False, allow_unicode=True)


def task_two_word(news):

    word1 = {}
    word2 = {}
    cnt_s = 0
    cnt_t = 0
    with open('./data/1word.pkl','rb') as f:
        word1 = pickle.load(f)

    for line in open(news, 'r', encoding='gbk', errors='ignore'):
        try:
            content = json.loads(line)
        except:
            continue
        info = content['html'] + content['title']
        info_len = len(info)
        seg = jieba.lcut(info, cut_all=False, HMM=True)
        # seg = [x for x in _seg if word1.get(x)]
        seg_len = len(seg)
        for i in range(seg_len - 1):
            key = None
            if i == 0 and word1.get(seg[0]):
                key = ''.join(['s_', seg[0]])
                cnt_s += 1
            elif i > 0 and word1.get(seg[i]) and not word1.get(seg[i - 1]):
                key = ''.join(['s_', seg[i]])
                cnt_s += 1
            elif word1.get(seg[i]) and not word1.get(seg[i + 1]):
                key = ''.join([seg[i], '_t'])
                cnt_t += 1
            elif i == seg_len - 2 and word1.get(seg[i + 1]):
                key = ''.join([seg[i + 1], '_t'])
                cnt_t += 1
            if key:
                word2[key] = word2.get(key, 0) + 1

            if word1.get(seg[i]) and word1.get(seg[i + 1]):
                key = ''.join([seg[i], '_', seg[i + 1]])
                word2[key] = word2.get(key, 0) + 1

    return (word2, cnt_s, cnt_t)


def two_word(rank):
    news = glob.glob('./training/sina_news_gbk/*.txt')
    word1 = {}
    word2 = {}
    cnt_s = 0
    cnt_t = 0
    with ProcessPoolExecutor() as executor:
        for results in executor.map(task_two_word, news):
            for key, value in results[0].items():
                word2[key] = word2.get(key, 0) + value
                cnt_s += results[1]
                cnt_t += results[2]

    with open('./data/1word.pkl','rb') as f:
        word1 = pickle.load(f)
    word1['s'] = cnt_s
    word1['t'] = cnt_t
    with open('./data/1word.pkl', 'wb') as f:
        pickle.dump(word1, f)
    # with open('./data/1word.yaml', 'w', encoding='utf-8') as f:
    #     yaml.dump(word1, f, default_flow_style=False, allow_unicode=True)

    # Select top rank% features for valid features
    word2 = dict(sorted(word2.items(), key = lambda x: x[1], reverse = True)[:int(len(word2)*rank)])

    if rank == 1.0:
        with open('./data/2word_whole.pkl', 'wb') as f:
            pickle.dump(word2, f)

        # with open('./data/2word_whole.yaml', 'w', encoding='utf-8') as f:
        #     yaml.dump(word2, f, default_flow_style=False, allow_unicode=True)
    else:
        with open('./data/2word.pkl', 'wb') as f:
            pickle.dump(word2, f)

        # with open('./data/2word.yaml', 'w', encoding='utf-8') as f:
        #     yaml.dump(word2, f, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='3c', type=str, choices=['2c', '3c', '2w'], help="Available models")
    parser.add_argument("--rank_3c", default=0.2, type=float, help="Select rank percent of chars data for predicting")
    parser.add_argument("--rank_2w", default=0.25, type=float, help="Select rank percent of words data for predicting")
    args = parser.parse_args()

    print('===> Preprocessing')
    start = datetime.datetime.now()
    if args.model_type == '2c':
        print("Model type: binary char")
        one_gram()
        two_gram()
    elif args.model_type == '3c':
        print("Model type: ternary char")
        one_gram()
        three_gram(rank=args.rank_3c)
    elif args.model_type == '2w':
        print("Model type: binary word")
        jieba.enable_parallel()
        # one_word()
        two_word(rank=args.rank_2w)
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed!')
    print('-' * 20)
