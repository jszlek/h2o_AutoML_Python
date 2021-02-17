#
# AutoML h2o script for testing model on external data
#

automl_model_name = 'GBM_1_AutoML_20210217_124842'                  # h2o model name
test_DF = '10cv_new_PLGA_FS_to_7_in_no1.csv'                                               # testing data frame

classification_problem: bool = False
run_regression_for_classification: bool = False

# ---------------------------------
# Load imports and start h2o server
# ---------------------------------

import os
import string
import random
import glob
import h2o
import csv
import sys
from h2o.automl import H2OAutoML
from pathlib import Path
from random import randint
import pandas as pd

# -------------------------------
# Import functions used
# -------------------------------
# -------------------------------
# Random key generator - function
# -------------------------------
def random_key_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# get current directory (PosixPath)
# -----------------------
my_current_dir = Path.cwd()

# get export directory and other subdirs (PosixPath)
# -----------------------
my_export_dir = my_current_dir.joinpath(str(my_current_dir) + '/export')

# check subdirectory structure
# ----------------------------------------
Path(my_export_dir).mkdir(parents=True, exist_ok=True)

# generate random port number
# -------------------------------
my_port_number = random.randint(54322,65000)

# Create three random strings
# -------------------------
aml_name = 'A' + random_key_generator(15) # for FS project name
aml2_name = 'A' + random_key_generator(15) # for classic approach project name
cluster_name = 'A' + random_key_generator(15) # for h2o cluster name

# -------------------------------------
# run h2o server
# -------------------------------------
h2o.init(nthreads=-1, 
         port=my_port_number,
         ice_root=str(my_export_dir),
         name=str(cluster_name),
         start_h2o=True)
# -------------------------------------

# ---------------------
# Load h2o autoML model
# ---------------------
aml3 = h2o.load_model(str(my_current_dir)+'/'+automl_model_name)

# -------------------
# Read dataset for h2o
# -------------------
data = pd.read_csv(test_DF, sep='\t', engine='python')
data_h2o = h2o.H2OFrame(data)


print('Model summary','\n', '-------------')
print('Algorithm: ' + aml3.algo)
print('Type: ' + aml3.type)
print('Run time [s]: ' + str(aml3.run_time))
print('Seed: ' + str(aml3.seed))

if aml3.rmse() != None:
    print('RMSE: ' + str(aml3.rmse()))
elif aml3.rmse() == None:
    print('RMSE: ' + 'None')

if aml3.mse() != None:
    print('MSE: ' + str(aml3.mse()))
elif aml3.mse() == None:
    print('MSE: ' + 'None')
    
if aml3.r2() != None:
    print('R2: ' + str(aml3.r2()))
elif aml3.r2() == None:
    print('R2: ' + 'None')
    
print('-------------')


aml3.summary()

if ('stackedensemble' in aml3.algo) is True:
    
    print('Metalearner name: ')
    print(aml3.metalearner().model_id, '\n')
    print('Metalearner algorithm: ')
    meta = h2o.get_model(aml3.metalearner().model_id)
    print(meta.algo, '\n')
    print('Ensemble contains models: ')
    
    model_list = []
    
    for model in aml3.params['base_models']['actual']:
        model_list.append(model['name'])
        
    for x in range(len(model_list)):
        print(model_list[x])
        
    print('---------------------------',"\n")


# make predictions
res_pred = aml3.predict(data_h2o)
pd_res_pred = res_pred.as_data_frame(use_pandas=True, header=True)

print('','\n')
print('','\n')
print('Model predictions: ')

# print out predictions
for i in range(len(pd_res_pred)):
    pd_res_pred.iloc[i][0]


# write predictions to res_pred.txt file
pd_res_pred.to_csv(path_or_buf = 'res_pred.txt' ,sep = '\t', header = True, index = False)

