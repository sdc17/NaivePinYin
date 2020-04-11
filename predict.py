import os
import ast
import math
import pickle
import argparse
from itertools import product
import numpy as np


def predict_two_char(ipath, opath, alpha=1e-10, st=25):
    # load gram
    gram1 = {}
    gram2 = {}
    pinyin = {}
    with open('./data/1gram.pkl', 'rb') as f:
        gram1 = pickle.load(f)
    with open('./data/2gram.pkl', 'rb') as f:
        gram2 = pickle.load(f)
    with open('./data/汉字拼音表.txt', encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            content = line.strip().split(' ')
            pinyin[content[0]] = content[1:]
    if not gram1 or not gram2 or not pinyin:
        print("Load gram error!")
        return 

    # predict
    with open(ipath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            probs = []
            traces = []
            words = line.strip().split(' ')
            pos = 0
            # fisrt char
            if words:
                first_cnt = {x : gram1[x] for x in pinyin[words[0].lower()]}
                tot = sum(first_cnt.values())
                first_prob = {x: y/tot for x, y in first_cnt.items()}
                # first_prob = {x: math.log(y/tot + 1e-30) for x, y in first_cnt.items()}
                prob = {}
                for key, value in first_prob.items():
                    prob[key] = (st * (1 - alpha) * gram2.get('s' + key, 0) / gram1['s'] + alpha * first_prob[key])

                # probs.append(first_prob)  
                probs.append(prob)

                trace = {x:'s' for x in pinyin[words[0].lower()]}
                traces.append(trace)

            # subsequent chars
            for i in range(1, len(words)):
    
                cnts = {x : gram1[x] for x in pinyin[words[i].lower()]}
                tot = sum(cnts.values())
                prob_one = {x: y/tot for x, y in cnts.items()}
                
                prob = {}
                trace = {}
                for j in pinyin[words[i]]:
                    dp = []
                    for k in pinyin[words[i - 1].lower()]:
                        dp.append(probs[i - 1][k] * ((1 - alpha) * gram2.get(k + j, 0) / (gram1[k] + 1e-30) + alpha * prob_one[j]))
                        # dp.append(probs[i - 1][k] + math.log((1 - alpha) * gram2.get(k + j, 0) / (gram1[k] + 1e-30) + alpha * prob_one[j] + 1e-30))
                    prob[j] = max(dp)
                    trace[j] = pinyin[words[i - 1].lower()][np.argmax(dp)]
                probs.append(prob)
                traces.append(trace)

            # last char
            for key in probs[-1].keys():
                probs[-1][key] *= (st * (1 - alpha) * gram2.get(key + 't', 0) / (gram1[key] + 1e-30) + alpha * 1.0)

            # backtrace
            last = max(probs[-1], key=lambda x:probs[-1][x])
            out = []
            out.append(last)
            for x in range(len(traces) - 1, 0, -1):
                out.append(traces[x][out[-1]])
            output = ''.join(out[::-1])
            with open(opath, 'a') as f:
                f.write(output + '\n')


def predict_three_char(ipath, opath, alpha=1e-10, st=40, full_model=False):
    # load gram
    gram1 = {}
    gram2 = {}
    gram3 = {}
    pinyin = {}
    with open('./data/1gram.pkl', 'rb') as f:
        gram1 = pickle.load(f)
    with open('./data/2gram.pkl', 'rb') as f:
        gram2 = pickle.load(f)    
    if full_model:
        with open('./data/3gram_whole.pkl', 'rb') as f:
            gram3 = pickle.load(f)
    else:
        with open('./data/3gram.pkl', 'rb') as f:
            gram3 = pickle.load(f)
    with open('./data/汉字拼音表.txt', encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            content = line.strip().split(' ')
            pinyin[content[0]] = content[1:]
    if not gram1 or not gram2 or not gram3 or not pinyin:
        print("Load gram error!")
        return 

    # predict
    with open(ipath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            probs = []
            traces = []
            words = line.strip().split(' ')

            # fisrt char
            if len(words) >= 1:
                first_cnt = {x : gram1[x] for x in pinyin[words[0].lower()]}
                tot = sum(first_cnt.values())
                first_prob = {x: y/tot for x, y in first_cnt.items()}
                # first_prob = {x: math.log(y/tot + 1e-30) for x, y in first_cnt.items()}
                prob = {}
                for key, value in first_prob.items():
                    prob[key] = (st * (1 - alpha) * gram2.get('s' + key, 0) / gram1['s'] + alpha * first_prob[key])

                # probs.append(first_prob)  
                probs.append(prob)

                trace = {x:'s' for x in pinyin[words[0].lower()]}
                traces.append(trace)

            # second char
            if len(words) >= 2:
                first_cnt = {x : gram1[x] for x in pinyin[words[0].lower()]}
                tot = sum(first_cnt.values())
                first_prob = {x: y/tot for x, y in first_cnt.items()}
                second_cnt = {x : gram1[x] for x in pinyin[words[1].lower()]}
                tot = sum(second_cnt.values())
                second_prob = {x : y/tot for x, y in second_cnt.items()}
                prob = {}
                trace = {}
                for key in second_prob.keys():
                    dp = [(st * (1 - alpha) * gram3.get('s' + x + key, 0) / (gram2.get('s' + x, 0) + 1e-30)) + alpha * second_prob[key] for x in first_prob.keys()]
                    prob[key] = max(dp)
                    trace[key] = pinyin[words[0].lower()][np.argmax(dp)]
                probs.append(prob)
                traces.append(trace)


            # subsequent chars
            for i in range(2, len(words)):
    
                cnts = {x : gram1[x] for x in pinyin[words[i].lower()]}
                tot = sum(cnts.values())
                prob_one = {x: y/tot for x, y in cnts.items()}
                
                prob = {}
                trace = {}
                for j in pinyin[words[i]]:
                    dp = []
                    for k in pinyin[words[i - 1].lower()]:
                        dp.append(max([probs[i - 1][k] * ((1 - alpha) * gram3.get(l + k + j, 0) / (gram2.get(l + k, 0) + 1e-30) + alpha * prob_one[j]) for l in pinyin[words[i - 2].lower()]]))
                    prob[j] = max(dp)
                    trace[j] = pinyin[words[i - 1].lower()][np.argmax(dp)]
                probs.append(prob)
                traces.append(trace)

            # last char
            if len(words) >= 2:
                for key in probs[-1].keys():
                    probs[-1][key] *= max([(st * (1 - alpha) * gram3.get(x + key + 't', 0) / (gram2.get(x + key, 0) + 1e-30) + alpha * 1.0) for x in probs[-2].keys()])
            else:
                for key in probs[-1].keys():
                    probs[-1][key] *= (st * (1 - alpha) * gram2.get(key + 't', 0) / (gram1[key] + 1e-30) + alpha * 1.0)

            # backtrace
            last = max(probs[-1], key=lambda x:probs[-1][x])
            out = []
            out.append(last)
            for x in range(len(traces) - 1, 0, -1):
                out.append(traces[x][out[-1]])
            output = ''.join(out[::-1])
            with open(opath, 'a') as f:
                f.write(output + '\n')


def predict_two_word(ipath, opath, alpha=1e-10, st=1, full_model=False):
    # load word
    word1 = {}
    word2 = {}
    pinyin = {}
    with open('./data/1word.pkl', 'rb') as f:
        word1 = pickle.load(f)
    if full_model:
        with open('./data/2word_whole.pkl', 'rb') as f:
            word2 = pickle.load(f)
    else:
        with open('./data/2word.pkl', 'rb') as f:
            word2 = pickle.load(f)
    with open('./data/汉字拼音表.txt', encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            content = line.strip().split(' ')
            pinyin[content[0]] = content[1:]
    if not word1 or not word2 or not pinyin:
        print("Load gram error!")
        return 

    # predict
    with open(ipath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            probs = []
            traces = []
            words = line.strip().split(' ')

            # fisrt word
            if len(words) == 1:
                first_cnt = {x : word1.get(x, 0) for x in pinyin[words[0].lower()]}
                tot = sum(first_cnt.values())
                first_prob = {x: y/tot for x, y in first_cnt.items()}
                prob = {}
                for key, value in first_prob.items():
                    prob[key] = (st * (1 - alpha) * word2.get('s_' + key, 0) / word1['s'] + alpha * first_prob[key])

                probs.append(prob)

                trace = {x:'s' for x in pinyin[words[0].lower()]}
                traces.append(trace)

            elif len(words) >= 2:
                first_cnt = {}
                for x in pinyin[words[0].lower()]:
                    for y in pinyin[words[1].lower()]:
                        if word1.get(x + y):
                            first_cnt[x + y] = word1[x + y]
                        # first_cnt[x + y] = word1.get(x + y, 0)
                tot = sum(first_cnt.values())
                first_prob = {x: y/tot for x, y in first_cnt.items()}
                prob = {}
                for key, value in first_prob.items():
                    prob[key] = (st * (1 - alpha) * word2.get('s_' + key, 0) / word1['s'] + alpha * first_prob[key])
                
                # fix zero prob
                if not prob:
                    fix_x = {}
                    fix_y = {}
                    for x in pinyin[words[0].lower()]: 
                        if word1.get(x):
                            fix_x[x] = word1[x]
                    for x in pinyin[words[1].lower()]:
                        if word1.get(x):
                            fix_y[x] = word1[x]
                    fix = max(fix_x, key=lambda x:fix_x[x]) + max(fix_y, key=lambda x:fix_y[x])
                    trace[fix] = 's'
                    prob[fix] = 1.0

                else: 
                    trace = {x : 's' for x in first_prob.keys()}

                probs.append(prob)
                traces.append(trace)

            if len(words) % 2 == 0:
                end = len(words)
            else:
                end = len(words) - 1

            # subsequent chars
            for i in range(2, end, 2):
                cnts = {}
                for x in pinyin[words[i].lower()]:
                    for y in pinyin[words[i + 1].lower()]:
                        if word1.get(x + y):
                            cnts[x + y] = word1[x + y]
                        # cnts[x + y] = word1.get(x + y, 0)
                tot = sum(cnts.values())
                prob_one = {x: y/tot for x, y in cnts.items()}
                
                prob = {}
                trace = {}
                
                for j in prob_one.keys():
                    dp = []
                    for m in probs[-1].keys():
                        dp.append(probs[i//2 - 1][m] * ((1 - alpha) * word2.get(m + '_' + j, 0) / (word1.get(m, 0) + 1e-60) + alpha * prob_one[j]))
                    prob[j] = max(dp)
                    trace[j] = list(probs[-1].keys())[np.argmax(dp)]
                
                # fix zero prob
                if not prob:
                    fix_x = {}
                    fix_y = {}
                    for x in pinyin[words[i].lower()]: 
                        if word1.get(x):
                            fix_x[x] = word1[x]
                    for x in pinyin[words[i + 1].lower()]:
                        if word1.get(x):
                            fix_y[x] = word1[x]
                    fix = max(fix_x, key=lambda x:fix_x[x]) + max(fix_y, key=lambda x:fix_y[x])
                    trace[fix] = max(probs[-1], key=lambda x: probs[-1][x])
                    prob[fix] = 1.0

                probs.append(prob)
                traces.append(trace)

            lenw = len(words)
            if lenw % 2 == 1 and lenw >= 3:
                cnts = {}
                for x in pinyin[words[lenw - 1].lower()]:
                    if word1.get(x):
                        cnts[x] = word1[x]
                    # cnts[x] = word1.get(x, 0)

                tot = sum(cnts.values())
                prob_one = {x: y/tot for x, y in cnts.items()}

                prob = {}
                trace = {}

                for i in prob_one.keys():
                    dp = []
                    for m in probs[-1].keys():
                        dp.append(probs[(lenw // 2) - 1][m] * ((1 - alpha) * word2.get(m + '_' + i, 0) / (word1.get(m, 0) + 1e-60) + alpha * prob_one[i]))
                    prob[i] = max(dp)
                    trace[i] = list(probs[-1].keys())[np.argmax(dp)]

                
                probs.append(prob)
                traces.append(trace)
        
            # for key in probs[-1].keys():
            #     probs[-1][key] *= (st * (1 - alpha) * word2.get(key + '_t', 0) / (word1.get(key, 0) + 1e-60) + alpha * 1.0)

            # backtrace
            last = max(probs[-1], key=lambda x:probs[-1][x])
            out = []
            out.append(last)
            for x in range(len(traces) - 1, 0, -1):
                out.append(traces[x][out[-1]])
            output = ''.join(out[::-1])
            with open(opath, 'a') as f:
                f.write(output + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default='./data/input.txt', type=str, help="input file")
    parser.add_argument("-o", default='./data/output.txt', type=str, help="output file")
    parser.add_argument("--model_type", default='3c', type=str, choices=['2c', '3c', '2w'], help="Available models")
    parser.add_argument("--full_model", default=False, type=ast.literal_eval, choices=[True, False], help="Use full model or not")
    
    args = parser.parse_args()

    if os.path.exists(args.i):
        if os.path.exists(args.o):
            os.remove(args.o)
        if args.model_type == '2c':
            predict_two_char(args.i, args.o, alpha=1e-10)
        elif args.model_type == '3c':
            predict_three_char(args.i, args.o, alpha=1e-10, full_model=args.full_model)
        elif args.model_type == '2w':
            predict_two_word(args.i, args.o, alpha=1e-10, full_model=args.full_model)
    else:
        print("Input file is invalid!")
