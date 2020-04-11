import os
import ast
import yaml
import glob
import datetime
import argparse
from predict import predict_two_char, predict_three_char, predict_two_word


def eval(ipath, model_type, record=False, full_model=False):
    input = []
    answer = []
    
    with open(ipath, 'r', encoding='gbk') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 0:
                input.append(line)
            else:
                answer.append(line)
    with open('./eval/input.txt', 'w') as f:
        for line in input:
            f.write(line)

    # with open('./eval/answer.txt', 'w') as f:
    #         for line in answer:
    #             f.write(line)

    if os.path.exists('./eval/output.txt'):
        os.remove('./eval/output.txt')

    if model_type == '2c':
        predict_two_char('./eval/input.txt', './eval/output.txt')
    elif model_type == '3c':
        predict_three_char('./eval/input.txt', './eval/output.txt', full_model=full_model)
    elif model_type == '2w':
        predict_two_word('./eval/input.txt', './eval/output.txt', full_model=full_model)

    cnt_word = 0
    cnt_word_correct = 0
    cnt_sen = len(answer)
    cnt_sen_correct = 0

    with open('./eval/output.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            length = len(line)
            cnt_word += length
            cnt_temp = 0
            for j in range(length):
                if line[j] == answer[i][j]:
                    cnt_word_correct += 1
                    cnt_temp += 1
            if cnt_temp == length:
                cnt_sen_correct += 1
    
    print("Character accuracy :{:.4f}, sentence accuracy: {:.4f}".format(cnt_word_correct/cnt_word, cnt_sen_correct/cnt_sen))
    
    # Record exp data
    if record:
        if not os.path.exists('./record'):
            os.mkdir('./record')
        records = glob.glob('./record/*.yaml')
        with open('./record/record_ver_' + str(len(records)) + '.yaml', 'w') as f:
            yaml.dump({'char_acc': cnt_word_correct/cnt_word, 'sen_acc': cnt_sen_correct/cnt_sen}, f, default_flow_style=False)      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default='./eval/eval.txt', type=str, help="input file")
    parser.add_argument("--record", default=True, type=ast.literal_eval, choices=[True, False], help="Record exp process")
    parser.add_argument("--model_type", default='3c', type=str, choices=['2c', '3c', '2w'], help="Available models")
    parser.add_argument("--full_model", default=False, type=ast.literal_eval, choices=[True, False], help="Use full model or not")
    args = parser.parse_args()

    if os.path.exists(args.i):
        print('===> Evaluating')
        start = datetime.datetime.now()
        eval(args.i, args.model_type, args.record, args.full_model)
        end = datetime.datetime.now()
        print('Time cost: {}'.format(end -start))
        print('===> Completed!')
        print('-' * 20)
    else:
        print("Input file is invalid!")

