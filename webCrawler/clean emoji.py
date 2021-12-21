import emoji
import re
import pandas as pd
import time

df = pd.read_excel("./google_rate/評論總表.xlsx", sheet_name="整體(剔除none, 奇怪符號, 英文評論)")

print(df.columns)

values = df["comment"].values

for value in values:
    text = emoji.demojize(value)
    result = re.sub(':\S+?:', '', text)
    df = df.replace({value: result})

cur_time = time.strftime("%Y%m%d_%H%M", time.localtime())

df.to_excel("./google_rate/output_{}.xlsx".format(cur_time), sheet_name="總表", engine='xlsxwriter')
