import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("imgs_logs/frontal _obs.csv")

data = data[0:1946] # Filtering until point where drone crashes (checked manually)
data = data[data.vel_x > 0] # Filtering first part where it's not moving
of_diff = data["Optical flow difference"]
div = data["Estimated divergence"]
of_diff = of_diff.unique()
div = div.unique()
of_diff = of_diff[30:]
div = div[30:]
plt.figure()
plt.plot(of_diff)
plt.figure()
plt.plot(div)
plt.show()