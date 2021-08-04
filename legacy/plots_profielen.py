# import sys
# import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

aansluiting_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/aansluiting_attributen.csv')
print(aansluiting_df.sort_values(by=['RND_ID']).head())

sub_aan = aansluiting_df[aansluiting_df['BASELOAD_PROFILE'] == 'PV']
print('subaan: ')
print(sub_aan)
print('length: ', sub_aan.size)

profielen_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/profielen.csv')

print(profielen_df.head())

PV = profielen_df["PV"]
DATUM_TIJDSTIP = profielen_df["DATUM_TIJDSTIP"]
WIND = profielen_df["WIND"]
KVKSEGMENT_1 = profielen_df["KVKSEGMENT_1"]
KO_LOGISTIEK = profielen_df["KO_LOGISTIEK"]

#plt.plot(DATUM_TIJDSTIP,PV)
#PV.plot()
#plt.show()

#KVKSEGMENT_1.plot()
#plt.show()
x = profielen_df["NUM"]
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x,KVKSEGMENT_1)
axs[0, 0].set_title('KVKSEGMENT_1')
axs[0, 1].plot(x, PV, 'tab:orange')
axs[0, 1].set_title('PV')
axs[1, 0].plot(x, WIND, 'tab:green')
axs[1, 0].set_title('WIND')
axs[1, 1].plot(x, KO_LOGISTIEK, 'tab:red')
axs[1, 1].set_title('KO_LOGISTIEK')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()




