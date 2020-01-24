import numpy as np
import pandas as pd
import pystan as ps
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#fetching data into table
import geopandas as gpd

My_file_path_name = r'C:\Users\himab\Documents\RiverSimilarity\data\HydroRIVERS_v10_na_shp\HydroRIVERS_v10_na_shp\HydroRIVERS_v10_na.dbf'

table = gpd.read_file(My_file_path_name)

table.rename(columns = {'LENGTH_KM':'l', 'UPLAND_SKM':'A', 'DIS_AV_CMS':'Q', 'ORD_STRA':'n'}, inplace = True)
table = table[['l','A','Q','n']]

# define the dataset
df = pd.DataFrame(table)
df = df.head(1000) #subsetting data to first 1000 data points
df['logQ'] = np.log(df['Q'])
df['logA']=np.log(df['A'])
# plot the dataset
df.plot(x="logA", y="logQ", kind="scatter", color="r", title="Dataset to analyse")
plt.show()

# Linear Regression with Stan
stan_model = ps.StanModel(file="LRmodel.txt")

data_dict = {"logQ": df["logQ"], "logA": df["logA"], "N": len(df)}

stan_fit = stan_model.sampling(data=data_dict)

# # extract the samples
stan_results = pd.DataFrame(stan_fit.extract())
print(stan_results.describe())

# here is one way to visualise the stan result, including uncertainty
# this does not include any filtering to eg 95%, it simply shows all inferences
for row in range(0, len(stan_results)):
    fit_line = np.poly1d([stan_results["a_one"][row], stan_results["a_zero"][row]])
    x = np.arange(6)
    y = fit_line(x)
    plt.plot(x, y, "b-", alpha=0.025, zorder=1)
plt.scatter(df["logA"], df["logQ"], c="r", zorder=2)
plt.title("Visualizing Stan Results")
plt.ylim([-5, 10])
plt.ylabel("logQ")
plt.xlim([-5, 10])
plt.xlabel("logA")
plt.show()
