'''
textファイルの日本語化
'''
# 日本語化したい拡張子.txtのファイル名をxに入れる
x = hoge
# 余談：f文字が標準化されるの？.format()は古い？
f = open(f"{x}.txt", "r", encoding="utf-8")
text = f.read()
f.close()
print(text)