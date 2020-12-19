import subprocess
from subprocess import PIPE

# 保存開始
# STEP1
# subprocess.check_call(['script', 'text.txt'])

# STEP2
# for i in range(1, 42):
#     try:
#         subprocess.check_call(['python', 'main.py', f'{i}'])
#     except:
#         print("subprocess.check_call() failed")

# STEP3
# result.pyを実行
subprocess.check_call(['python', 'result.py'])

# STEP4
# ターミナルの出力を.txtファイルに保存
subprocess.check_call(['exit'])
