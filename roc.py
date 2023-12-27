from sklearn.metrics import roc_curve, auc

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
label=np.load("UCSDped2_gt.npy",allow_pickle=True)
print(label)
import seaborn as sns
# sns.set(style="white")
# plt.style.use('ggplot')
lables=[]
for i in label:
    lables.append(i[8:-7])

after_labels=np.concatenate(lables,axis=0)

data1=np.load("score_after0.npy")

data2=np.load("score_after1.npy")

data3=np.load("score_after6.npy")

# data4=np.load("score_after258.npy")
#
# data5=np.load("score_after28.npy")
#
# data6=np.load("score_after195.npy")
#
# data7=np.load("score_after585.npy")

fpr, tpr, thresholds = metrics.roc_curve(after_labels, data1, pos_label=0)

fpr2, tpr2, thresholds2=metrics.roc_curve(after_labels,data2,pos_label=0)

fpr3, tpr3, thresholds3=metrics.roc_curve(after_labels,data3,pos_label=0)

# fpr4, tpr4, thresholds4=metrics.roc_curve(after_labels,data4,pos_label=0)
#
# fpr5, tpr5, thresholds5=metrics.roc_curve(after_labels,data5,pos_label=0)
#
# fpr6, tpr6, thresholds6=metrics.roc_curve(after_labels,data6,pos_label=0)
#
# fpr7, tpr7, thresholds7=metrics.roc_curve(after_labels,data7,pos_label=0)

print(fpr, tpr,thresholds)

print("auc",metrics.auc(fpr,tpr))

plt.figure()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
lw = 2
plt.plot(fpr, tpr,sns.xkcd_rgb["cherry"],lw=2,
          label='本文模型-ROC'),

print("本文-ROC",metrics.auc(fpr,tpr))
plt.plot(fpr2, tpr2, color='blue',lw=2,
          label='文献[18]-ROC')


# plt.plot(fpr4, tpr4, color='teal',lw=2,
#           label='文献[21]-ROC')



plt.plot(fpr3, tpr3, color='orange',lw=2,
          label='文献[20]-ROC')
print("文献[10]-ROC",metrics.auc(fpr3,tpr3))

# print("文献[14]-ROC",metrics.auc(fpr4,tpr4))
# plt.plot(fpr5, tpr5, color='teal',lw=2,
#           label='模型一')
#
# plt.plot(fpr6, tpr6, color='red',lw=2,
#           label='模型二' )
#
# plt.plot(fpr, tpr, color='darkorange',lw=2,
#           label='模型三')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("./roc3.png")
plt.figure(figsize=(5,5),dpi=600)

plt.show()
