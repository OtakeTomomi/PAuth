'''
textファイルの日本語化
'''

f = open("hoge.txt","r",encoding="utf-8")
text = f.read()
f.close()
print(text)