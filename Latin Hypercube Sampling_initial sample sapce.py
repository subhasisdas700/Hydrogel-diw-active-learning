import numpy as np
import pandas as pd
from scipy.stats import qmc

ranges = {
    'Alginate': (0.5, 4),
    'PVA': (5, 15),
    'Glycerol': (0.5, 10),
    'Gelatin': (5, 25)
}
n_components = len(ranges)
n_samples = 10

# Use SciPy's LatinHypercube
sampler = qmc.LatinHypercube(d=n_components)
lhs_samples = sampler.random(n=n_samples)

names = list(ranges.keys())
scaled = np.zeros_like(lhs_samples)
for i, name in enumerate(names):
    low, high = ranges[name]
    scaled[:, i] = lhs_samples[:, i] * (high - low) + low

# Calculate Water so that sum to 100
comps = []
for row in scaled:
    alg, pva, gly, gel = row
    water = 100 - (alg + pva + gly + gel)
    if 0 <= water <= 100:
        comps.append([alg, pva, gly, gel, water])

comps = np.array(comps)
print(f"Generated {len(comps)} valid samples.")

df_samples = pd.DataFrame(comps, columns=['Alginate', 'PVA', 'Glycerol', 'Gelatin', 'Water'])
print(df_samples)
df_samples.to_excel("initial_design_samples.xlsx", index=False)
