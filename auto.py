import subprocess, sys
from subprocess import PIPE

# proc = subprocess.run("date", shell=True, stdout=PIPE, stderr=PIPE, text=True)
# date = proc.stdout
# print('STDOUT: {}'.format(date))
for i in range(1,42):
    try:
        subprocess.check_call(['python','main.py',f'{i}'])
    except:
        print("subprocess.check_call() failed")

# subprocess.run(['23'], input='23', encoding='UTF-8')
