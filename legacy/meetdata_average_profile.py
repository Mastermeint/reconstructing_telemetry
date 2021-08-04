from numpy.core.function_base import linspace
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/gv_meetdata_select.csv', nrows=1000)
print(meetdata_df.head())
#meetdata_df.T.plot()
#plt.show()
shape = np.shape(meetdata_df)
shape = np.asarray(shape)

meetdata_df = np.asarray(meetdata_df.fillna(0))

average_profile = np.zeros(shape[1])

for i in range(shape[1]):
    average_profile[i] = np.sum(meetdata_df[:,i])/1000

print(average_profile)
plt.plot(average_profile[1:])
plt.ylabel('Energy Consumption')
plt.xlabel('Jan - Feb - Mar - Apr - May - Jun - Jul - Aug - Sep - Oct - Nov - Dec')
plt.title('Large - Intermediate average profile consumption')
plt.legend()
plt.show()
