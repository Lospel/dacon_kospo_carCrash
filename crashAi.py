#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

df_train = pd.read_csv("./open/train.csv")
df_test = pd.read_csv("./open/test.csv")

print(df_train.head())

#%%
# 결측치 테스트
import missingno as msno
msno.bar(df_train, fontsize=15, color="red")

# %%
df_train["label"].unique()
# %%
df_train["label"].value_counts(dropna=False).sort_index()
# %%
# train 데이터 시각화
label = np.unique(df_train["label"])
counts = np.array(df_train["label"].value_counts().sort_index())
plt.bar(label, counts)
plt.xlabel("Label")
plt.ylabel("counts")
plt.xticks(label)
plt.show()
# %%
# Label Info.
# 13가지의 차량 충돌 상황 Class의 세부 정보
# crash : 차량 충돌 여부 (No/Yes)
# ego-Involve : 본인 차량의 충돌 사고 연류 여부 (No/Yes)
# weather : 날씨 상황 (Normal/Snowy/Rainy)
# timing : 낮과 밤 (Day/Night)
# ego-Involve, weather, timing의 정보는 '차량 충돌 사고'가 일어난 경우에만 분석합니다.
crash = []
ego_Involve = []
weather = []
timing = []

#crash
for x in df_train["label"]:
    if x == 0:
        crash.append(0) #crash No
    else:
        crash.append(1) #crash Yes
df_train["crash"] = crash

#ego_Involve
for x in df_train["label"]:
    if x == 0:
        ego_Involve.append(-1) #Null
    elif x<7:
        ego_Involve.append(1) #ego_Involve Yes
    else:
        ego_Involve.append(0) #ego_Involve No
df_train["ego_Involve"] = ego_Involve

#weather
for x in df_train["label"]:
    if x==0:
        weather.append(-1) #Null
    elif x==(1 or 2 or 7 or 8):
        weather.append(0) #normal
    elif x==(3 or 4 or 9 or 10):
        weather.append(1) #snowy
    else:
        weather.append(2) #rainy
df_train["weather"]=weather

#timing
for x in df_train["label"]:
    if x==0:
        timing.append(-1)
    elif x%2==1:
        timing.append(0) #day
    else:
        timing.append(1) #night
df_train["timing"]=timing
     
print(df_train.head())
# %%
crash_counts = df_train["crash"].value_counts(dropna=False).sort_index()
ego_counts = df_train["ego_Involve"].value_counts(dropna=False).sort_index()
weather_counts = df_train["weather"].value_counts(dropna=False).sort_index()
timing_counts = df_train["timing"].value_counts(dropna=False).sort_index()
print(crash_counts, ego_counts, weather_counts, timing_counts)
# %%
# Crash 비교 그래프 
# 사고 비율이 약 2배 정도 차이나는 것을 알 수 있음
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)
ax.bar([0,1], crash_counts)
plt.xlabel("crash")
plt.ylabel("counts")
ax.set_xticks([0,1])
ax.set_xticklabels(['No','Yes'])
plt.title("Crash")
plt.show()
# %%
# ego_Involve 비교 그래프 
# 본인 차량의 사고 연류 여부는 거의 비슷한 것을 알 수 있음
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)
ax.bar([-1,0,1], ego_counts)
plt.xlabel("Ego-Involve")
plt.ylabel("counts")
ax.set_xticks([-1,0,1])
ax.set_xticklabels(['No_Crash','No','Yes'])
plt.title("Ego-Involve")
plt.show()
# %%
# weather 비교 그래프 
# 눈 오는 날에는 사고가 가장 적고, 비 오는 날에는 사고가 가장 많은 것을 알 수 있음
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)
ax.bar([-1,0,1,2], weather_counts)
plt.xlabel("Weather")
plt.ylabel("counts")
ax.set_xticks([-1,0,1,2])
ax.set_xticklabels(['No_Crash','Normal','Snowy','Rainy'])
plt.title("Weather")
plt.show()
# %%
# timing 비교 그래프 
# 사고는 밤보다 낮 시간에 가장 많이 일어나는 것을 알 수 있음
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)
ax.bar([-1,0,1], timing_counts)
plt.xlabel("Timing")
plt.ylabel("counts")
ax.set_xticks([-1,0,1])
ax.set_xticklabels(['No_Crash','Day','Night'])
plt.title("Timing")
plt.show()
# %%
df_train.to_csv('train_label_info.csv',index=False)
# %%
