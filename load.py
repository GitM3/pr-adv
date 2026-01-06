from phenobench import PhenoBench
import matplotlib.pyplot as plt
from phenobench.visualization import draw_semantics
import numpy as np
import os

from pprint import pprint


train_data = PhenoBench("~/Development/08_ADV/PhenoBench", 
                        target_types=["semantics"])


print(
    f"PhenoBench ({train_data.split} split) contains {len(train_data)} images. We loaded the following targets: {train_data.target_types}."
)
print("The first entry contains the following fields:")
pprint([f"{k} -> {type(v)}" for k, v in train_data[0].items()])

n_samples = 4
n_rows = 4
fig, axes = plt.subplots(ncols=n_samples, nrows=n_rows, figsize=(3 * n_samples, 3 * n_rows))

indexes = np.random.choice(len(train_data), n_samples)

for i in range(n_rows):
    for j in range(n_samples):
        axes[i, j].set_axis_off()

for id, idx in enumerate(indexes):
    axes[0, id].set_title(os.path.splitext(train_data[idx]["image_name"])[0])
    draw_semantics(axes[0, id], train_data[idx]["image"], train_data[idx]["semantics"], alpha=0.5)

plt.show()
