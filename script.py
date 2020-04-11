import os


# 2 char
# command = 'python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=2c'

# 3 char for 20% model
# command = 'python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=3c --full_model=False'

# 3 char for full model
command = 'python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=3c --full_model=True'

# 2 word for 20% model
# command = 'python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=2w --full_model=False'

# 2 word for full model
# command = 'python predict.py -i=./data/input.txt -o=./data/output.txt --model_type=2w --full_model=True'




# TEST 2 char
# command = 'python eval.py -i=./eval/eval.txt --record=False --model_type=2c'

# TEST 3 char for 20% model
# command = 'python eval.py -i=./eval/eval.txt --record=False --model_type=3c --full_model=False'

# TEST 3 char for full model
# command = 'python eval.py -i=./eval/eval.txt --record=False --model_type=3c --full_model=True'

# TEST 2 word for 20% model
# command = 'python eval.py -i=./eval/eval.txt --record=False --model_type=2w --full_model=False'

# TEST 2 word for 100% model
# command = 'python eval.py -i=./eval/eval.txt --record=False --model_type=2w --full_model=True'

os.system(command)

