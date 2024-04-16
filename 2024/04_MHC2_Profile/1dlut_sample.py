from colour.models import eotf_ST2084, eotf_inverse_ST2084
import numpy as np
import matplotlib.pylab as plt

num_of_sample = 256
gain = 0.5
step1 = np.linspace(0, 1, num_of_sample)
step2 = eotf_ST2084(step1) * gain
step3 = eotf_inverse_ST2084(step2)

plt.plot(step1, step3)
plt.grid(True)  # Enable grid
plt.xlim(0.0, 1.0)  # Set x-axis range
plt.ylim(0.0, 1.0)  # Set y-axis range
plt.savefig("./out.png")  # Save the plot
plt.show()
