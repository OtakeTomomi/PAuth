import subprocess
from subprocess import PIPE

# 保存開始
# subprocess.check_call(['script','text.txt'])

for i in range(1,2):
    try:
        subprocess.check_call(['python','main.py',f'{i}'])
    except:
        print("subprocess.check_call() failed")

# ターミナルの出力を.txtファイルに保存
# subprocess.check_call(['exit'])


# result.pyを実行
# subprocess.check_call(['python','result.py'])
