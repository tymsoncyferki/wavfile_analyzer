from scipy.io import wavfile
from analysis import FileAnalyzer

# rate, data = wavfile.read("file_example_WAV_5MG.wav")

# rate -> ilosc sygnałów na sekunde
# data -> sygnały

# print(rate)

# print(data.shape)
# print(data)

# length = data.shape[0] / rate

import matplotlib.pyplot as plt
import numpy as np

analyzer = FileAnalyzer("file_example_WAV_5MG.wav")

# silence, time = analyzer.silence()
# sr = analyzer.silent_ratio()
# volume, _ = analyzer.volume()

# time_filtered = time[silence==1]
# zeros = np.zeros(shape=len(time_filtered))

# cor, time = analyzer.autocorrelation(100)

# # print(cor)
# plt.figure(figsize=(10, 4))
# # plt.scatter(time_filtered, zeros, label="Silence", c='red', marker=2, s=50)
# plt.plot(time, cor, label="Volume", alpha=0.3)
# plt.xlabel("Czas [s]")
# plt.ylabel("Cisza")
# plt.title(f"Cisza na poziomie ramek, SR={sr:.4f}")
# plt.legend()
# plt.show()


energy, _ = analyzer.ste()
avSTE = np.sum(energy) / analyzer.length
print(energy[0])
print(np.sum([(np.sign(0.5 * avSTE - ste)) + 1 for ste in energy]))  

# print(analyzer.lster())
print(avSTE)