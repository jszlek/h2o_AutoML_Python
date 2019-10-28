#
# AutoML h2o script for testing model on external data
#

automl_model_name = 'GBM_grid_1_AutoML_20191028_221808_model_37'                  # h2o model name
test_DF = 'PLGA_300in_SR_BAZA.txt'                                               # testing data frame

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
import statistics
import psutil
import csv
import sys
from h2o.automl import H2OAutoML
from pathlib import Path
from random import randint
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from copy import copy
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold # import KFold
from sklearn import preprocessing
import matplotlib
# Force matplotlib to not use any X window backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
         ice_root=str(my_current_dir),
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



#if (classification_problem && !run_regression_for_classification){
      #test.hex[,ncol(test.hex)]<-as.factor(test.hex[,ncol(test.hex)])    
#}


print('Model summary','\n', '-------------')
print('Algorithm: ' + aml3.algo)
print('Type: ' + aml3.type)
print('Run time [s]: ' + str(aml3.run_time))
print('Seed: ' + str(aml3.seed))
print('RMSE: ' + str(aml3.rmse()))
print('R2: ' + str(aml3.r2()) + '\n')

h2o.summary(aml3)

if ("stackedensemble" in aml3.algo) is True:
    
    print("Metalearner name: ","\n")
    cat(aml3@model$metalearner$name,"\n")
    cat("","\n")
    cat("","\n")
    cat("Ensemble contains models:","\n")
    ensemble.models.names <- sapply(aml3@parameters$base_models, "[[", "name")
    print(ensemble.models.names)
    cat("","\n")
    cat("","\n")
    
}


cat("Model performance","\n")
cat("RMSE: ", aml3@model$cross_validation_metrics@metrics$RMSE, "\n")
cat("MSE: ", aml3@model$cross_validation_metrics@metrics$MSE, "\n")
cat("R2: ", aml3@model$cross_validation_metrics@metrics$r2, "\n")
cat("AIC: ", aml3@model$cross_validation_metrics@metrics$AIC, "\n")

















# get export directory and other subdirs (PosixPath)
# -----------------------
my_export_dir = my_current_dir.joinpath(str(my_current_dir) + '/export')
my_10cv_FS_dir = my_current_dir.joinpath(str(my_current_dir) + '/10cv_FS')
my_10cv_orig_dir = my_current_dir.joinpath(str(my_current_dir) + '/10cv_orig')

# check subdirectory structure
# ----------------------------------------
Path(my_export_dir).mkdir(parents=True, exist_ok=True)
Path(my_10cv_FS_dir).mkdir(parents=True, exist_ok=True)
Path(my_10cv_orig_dir).mkdir(parents=True, exist_ok=True)


# check runtime mode - either many servers on the machine (server_multicore = F) or one server per one machine (server_multicore = T)
# -------------------------------------------
if server_multicore is True:
    my_cores = 1
else:
    my_cores = psutil.cpu_count()

# check system free mem and apply it to the server
# ------------------------------------------------
memfree = psutil.virtual_memory().total
memfree_g = int(round(memfree/1024/1024/1024/my_cores,3))

if memfree_g < 2:
 memfree_g = 2

if my_max_ram_allowed > 0:
  memfree_g = my_max_ram_allowed

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
h2o.init(nthreads=my_threads, 
         min_mem_size=memfree_g,
         max_mem_size=memfree_g,
         port=my_port_number,
         ice_root=str(my_export_dir),
         name=str(cluster_name),
         start_h2o=True)
# -------------------------------------





h2o.init()


# make predictions
res_pred<-as.data.frame(h2o.predict(aml3,test.hex))[,1]

cat("","\n")
cat("","\n")
cat("Model predictions:","\n")
as.data.frame(res_pred)

# write predictions to res_pred.txt file
write.table(as.data.frame(res_pred), file="res_pred.txt", row.names=FALSE)
