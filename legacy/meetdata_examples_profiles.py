import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/gv_meetdata_select.csv', nrows=11)
print(meetdata_df.head())
meetdata_df.T.plot()
plt.show()

shape = np.shape(meetdata_df)
shape = np.asarray(shape)

meetdata_df = np.asarray(meetdata_df.fillna(0))

first_company = meetdata_df[1,:]
second_company = meetdata_df[2,:]
third_company = meetdata_df[3,:]
fourth_company = meetdata_df[4,:]
fifth_company = meetdata_df[5,:]
sixth_company = meetdata_df[6,:]
seventh_company = meetdata_df[7,:]
eighth_company = meetdata_df[8,:]
nineth_company = meetdata_df[9,:]

x = np.arange(0,shape[1],1)
fig, axs = plt.subplots(3, 3)
axs[0, 0].plot(x,first_company)
axs[0, 0].set_title('first_company')
axs[0, 1].plot(x, second_company, 'tab:orange')
axs[0, 1].set_title('second_company')
axs[0, 2].plot(x, third_company, 'tab:green')
axs[0, 2].set_title('third_company')
axs[1, 0].plot(x, fourth_company, 'tab:red')
axs[1, 0].set_title('fourth_company')
axs[1, 1].plot(x, fifth_company, 'tab:red')
axs[1, 1].set_title('fifth_company')
axs[1, 2].plot(x, sixth_company, 'tab:red')
axs[1, 2].set_title('sixth_company')
axs[2, 0].plot(x, seventh_company, 'tab:red')
axs[2, 0].set_title('seventh_company')
axs[2, 1].plot(x, eighth_company, 'tab:red')
axs[2, 1].set_title('eighth_company')
axs[2, 2].plot(x, nineth_company, 'tab:red')
axs[2, 2].set_title('nineth_company')


for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()