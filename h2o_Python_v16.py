# upper part

server_multicore = True
my_threads = 6
my_max_ram_allowed = 0
my_keep_cross_validation_predictions = True
my_keep_cross_validation_models = True
my_keep_cross_validation_fold_assignment = True
skel_plik = '10cv_new_PLGA_FS_to_16_in_no'  # Please provide n-fold training core filenames if use_classic_approach is
# True (filenames MUST contain '_no' part), copy files to ./10cv_FS
skel_plik1 = 't-10cv_new_PLGA_FS_to_16_in_no' # Please provide n-fold testing core filenames if use_classic_approach is True (filenames MUST contain '_no' part), copy files to ./10cv_FS
fs_data = 'PLGA_300in_SR_BAZA.txt' # Please provide full filename if perform_FS is True or classic_approach without n-fold cv are to be run

# user specified seeds
my_seed_FS = 1
my_seed_classic_approach = 1
my_seed_FS_10cv = 1

# backward compatibility - if true - a classic way of training AutoML will be performed
# YOU NEED TO COPY t-res files into ./10cv_FS folder - folds have to numbered t-res*no1.txt, t-res*no2.txt etc. every file must have header in first row!
use_classic_approach: bool = True

# save pojo or mojo model boolean = True/False
save_pojo_or_mojo: bool = True

# How many fold in cross validation is used only if perform_FS is True
no_folds = 10

# 10cv AutoML execution time
# Refers to classic_approach and perform_FS
h2o_max_runtime_secs_10cv = 15
h2o_max_runtime_secs_2nd_time_10cv = 5*60

# How many short loops of 10cv
# - only if perform_FS is True
my_10cv_loops = 30

# perfor_FS AutoML execution time
# - only if perform_FS is True
FS_h2o_max_runtime_secs = 15
FS_h2o_max_runtime_secs_2nd_time = 1*60

# How many loops of FS
# - only if perform_FS is True
my_FS_loops = 30

# -----------------------------------------------------------
# Perform feature selection before training AutoML in 10-fold cv
# Options:

# Main option True/False
perform_FS: bool = False

# Scale by original score or rmse - this is only for comparison with fscaret - set False to scale by the RMSE
# - only if perform_FS is True
original_scale: bool = True

# create core file name - used only when perform_FS is True to create filenames
core_filename = 'new_PLGA'

# Manually include features, eg. 'Time_min' in dissolution profile 
# - used only when perform_FS is True
include_features = []

# Feature selection threshold - range = [0; 1] - usually between 0.01 and 0.001
# - only if perform_FS is True
fs_threshold = 0.01

# Feature selection short loop RMSE threshold
# - only if perform_FS is True
rmse_fs_short_loop_threshold = 15.0

# 10-cv short loop RMSE threshold
# - only if perform_FS is True
rmse_10cv_short_loop_threshold = 15.0

# Which column contains indicies to make split - 1 = 1st col, 2 = 2nd col etc. 
# - only if perform_FS is True
index_column = 1


# down part

# ------------------------------------------------------------------------------
# Do not modify below
# ------------------------------------------------------------------------------

# IMPORTS

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
import re
# Force matplotlib to not use any X window backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Read dataset for FS
data = pd.read_csv(fs_data, sep='\t', engine='python')

# -------------------------------
# Random key generator - function
# -------------------------------
def random_key_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# -------------------------------------------------------------------------------

# get current directory (PosixPath)
# -----------------------
my_current_dir = Path.cwd()

# get export directory and other subdirs (PosixPath)
# -----------------------
my_export_dir = my_current_dir.joinpath(str(my_current_dir) + '/export')
my_10cv_FS_dir = my_current_dir.joinpath(str(my_current_dir) + '/10cv_FS')
my_10cv_orig_dir = my_current_dir.joinpath(str(my_current_dir) + '/10cv_orig')
my_test_external = my_current_dir.joinpath(str(my_current_dir) + '/test_external')
my_pojo_or_mojo_FS = my_current_dir.joinpath(str(my_current_dir) + '/pojo_or_mojo_FS')
my_pojo_or_mojo_10cv = my_current_dir.joinpath(str(my_current_dir) + '/pojo_or_mojo_10cv')
my_model_FS = my_current_dir.joinpath(str(my_current_dir) + '/model_FS')
my_model_10cv = my_current_dir.joinpath(str(my_current_dir) + '/model_10cv')


# check subdirectory structure
# ----------------------------------------
Path(my_export_dir).mkdir(parents=True, exist_ok=True)
Path(my_10cv_FS_dir).mkdir(parents=True, exist_ok=True)
Path(my_10cv_orig_dir).mkdir(parents=True, exist_ok=True)
Path(my_test_external).mkdir(parents=True, exist_ok=True)
Path(my_pojo_or_mojo_FS).mkdir(parents=True, exist_ok=True)
Path(my_pojo_or_mojo_10cv).mkdir(parents=True, exist_ok=True)
Path(my_model_FS).mkdir(parents=True, exist_ok=True)
Path(my_model_10cv).mkdir(parents=True, exist_ok=True)


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

# --------------------------------------
# 1st option - perform_FS
# --------------------------------------

if perform_FS is True:
    
    # checking if my_10cv_FS_dir, my_10cv_orig_dir, my_pojo_or_mojo_FS, my_pojo_or_mojo_10cv, my_model_FS, my_model_10cv are empty if not delete content
    print('\n' + 'Checking for non-empty dirs ...' + '\n')
    checking_list = [my_10cv_FS_dir, my_10cv_orig_dir, my_pojo_or_mojo_FS, my_pojo_or_mojo_10cv, my_model_FS, my_model_10cv]
    for checked_dir in checking_list:
        if len(os.listdir(checked_dir)) > 0:
            print('Removing files from ' + str(checked_dir) + ':')
            files_to_remove = glob.glob(str(checked_dir.joinpath(str(checked_dir)+'/*')))
            for f in files_to_remove:
                print(str(f))
                os.remove(f)
    
    # divide between X(input) and y(output)
    # First column contains group indicies
    # Last column contains output
    ncols = data.shape[1]-1
    nrows = data.shape[0]
    
    X = data.drop(data.columns[[0,ncols]], axis=1)
    y = data[data.columns[ncols]]
    
    # needed to make cv by groups - first column contains indicies!
    groups = data[data.columns[[index_column - 1]]]
    
    # Define FS_loops counter
    no_FS_loops = 1
    
    # the counter is set from 1, therefore = my_FS_loops + 1
    while no_FS_loops < (my_FS_loops + 1): 
        
        # split on train - test dataset by group 'Formulation no' - this is for Feature Selection
        tmp_train_inds, tmp_test_inds = next(GroupShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3).split(X, groups=groups))
        tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = X.iloc[tmp_train_inds], X.iloc[tmp_test_inds], y.iloc[tmp_train_inds], y.iloc[tmp_test_inds]
        
        # Write splits on disk
        tmp_train_set = pd.concat([tmp_X_train, tmp_y_train], axis=1)
        tmp_test_set = pd.concat([tmp_X_test, tmp_y_test], axis=1)
        
        tmp_y_idx = tmp_train_set.columns[tmp_train_set.shape[1]-1]
        
        tmp_training_frame = h2o.H2OFrame(tmp_train_set)
        tmp_testing_frame = h2o.H2OFrame(tmp_test_set)
        
        # print out no of loop
        print('\n' + 'Starting FS loop no: ' + str(no_FS_loops) + '\n')
        tmp_my_random_seed_FS = random.randint(1,100000000)
        print('Temp random seed: ' + str(tmp_my_random_seed_FS) + '\n')
        
        tmp_aml_name = 'A' + random_key_generator(15)
        
        # autoML settings
        tmp_FS_model = H2OAutoML(max_runtime_secs = FS_h2o_max_runtime_secs,
                                 seed = tmp_my_random_seed_FS,
                                 project_name = tmp_aml_name,
                                 export_checkpoints_dir = str(my_export_dir),
                                 keep_cross_validation_models = my_keep_cross_validation_models,
                                 keep_cross_validation_predictions = my_keep_cross_validation_predictions,
                                 keep_cross_validation_fold_assignment = my_keep_cross_validation_fold_assignment,
                                 verbosity = 'info',
                                 sort_metric = 'RMSE')
        
        # train model for FS
        tmp_FS_model.train(y = tmp_y_idx, training_frame = tmp_training_frame, leaderboard_frame = tmp_testing_frame)
        
        # write first model rmse metrics
        if no_FS_loops is 1:
            tmp_FS_rmse = tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE']
        
        # print out RMSE for the model
        print('\n' + 'RMSE for FS loop no: ' + str(no_FS_loops) + ' is ' + str(tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE']) + '\n')
        
        # if new tmp_FS_model has better performance overwrite it to aml
        if tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE'] < tmp_FS_rmse:
            
            # overwrite rmse for the tmp_FS_model - the leader
            tmp_FS_rmse = tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE']
            
            #generate an unique file name based on the id and record
            file_name_train = str(core_filename)+"_h2o_train_for_FS"+".txt"
            file_name_test = str(core_filename)+"_h2o_test_for_FS"+".txt"
            
            tmp_train_set.to_csv(file_name_train, index=False, sep="\t")
            tmp_test_set.to_csv(file_name_test, index = False, sep="\t")
            
            y_idx = tmp_y_idx
            
            training_frame = tmp_training_frame
            testing_frame = tmp_testing_frame
            
            my_random_seed_FS = tmp_my_random_seed_FS
            aml_name = tmp_aml_name
            
            print('Current best aml name: ' + str(aml_name))
            print('Current best seed: ' + str(my_random_seed_FS) + '\n')
        
            # if new tmp_FS_model RMSE is lower or equal has better performance overwrite it to aml
            if tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE'] <= rmse_fs_short_loop_threshold:

                print('\n' + 'Performance of obtained model is better than set threshold: ' + '\n')
                print('Threshold was set to: ' + str(rmse_fs_short_loop_threshold) + '\n')
                print('Performance of obtained model is: ' + str(tmp_FS_rmse) + '\n')
                print('Breaking the short FS loop')

                # Making no_FS_loops equal to my_FS_loops to break the while loop
                no_FS_loops = my_FS_loops 
            
        
        # FS_loop counter +1
        no_FS_loops += 1
    
    # Once again perform FS on 'the best' train / test dataset, but this time for much longer
    
    print('\n' + 'Used best aml name: ' + str(aml_name))
    print('Used best seed: ' + str(my_random_seed_FS) + '\n')
    
    # autoML settings
    aml = H2OAutoML(max_runtime_secs = FS_h2o_max_runtime_secs_2nd_time,
                    seed = my_random_seed_FS,
                    project_name = aml_name,
                    export_checkpoints_dir = str(my_export_dir),
                    keep_cross_validation_models = my_keep_cross_validation_models,
                    keep_cross_validation_predictions = my_keep_cross_validation_predictions,
                    keep_cross_validation_fold_assignment = my_keep_cross_validation_fold_assignment,
                    verbosity = 'info',
                    sort_metric = 'RMSE')
        
    # train model for FS
    aml.train(y = y_idx, training_frame = training_frame, leaderboard_frame = testing_frame)
    
    # saving model
    my_model_FS_path = h2o.save_model(aml.leader, path = './model_FS')
    
    print('')
    print('Final model of feature selection is located at: ')
    print(str(my_model_FS_path))
    print('')
    
    # Download POJO or MOJO
    if save_pojo_or_mojo is True:
        if aml.leader.have_pojo is True:
            aml.leader.download_pojo(get_genmodel_jar = True, path = './pojo_or_mojo_FS')
        if aml.leader.have_mojo is True:
            aml.leader.download_mojo(get_genmodel_jar = True, path = './pojo_or_mojo_FS')
    
    # get leader model key
    model_key = aml.leader.key
    
    lb = aml.leaderboard
    lbdf = lb.as_data_frame()
    
    print("Leaderboard: ","\n")
    lbdf.head()
    
    
    if ("StackedEnsemble" in model_key) is False:
        # get varimp_df
        varimp_df = aml.leader.varimp(use_pandas = True).iloc[:,[0,2]]
        scaled_var_imp_df = varimp_df
        
        # Drop variables by a fs_threshold condition
        scaled_var_imp_df = scaled_var_imp_df[scaled_var_imp_df.scaled_importance > fs_threshold]
        
        # Sort by 'scaled_importance' values
        scaled_var_imp_df_sorted = scaled_var_imp_df.sort_values(by=['scaled_importance'], ascending = False)
        
        # Plot and save bar chart
        plt.rcParams['xtick.labelsize'] = 4
        ax = scaled_var_imp_df_sorted.plot.bar(y='scaled_importance', x = 'variable', rot=90)
        plt.tight_layout()
        plt.savefig('FS_result_h2o.pdf', format='pdf', dpi=1200)
        
        
    else:
        model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
        
        # get the best model key
        m = h2o.get_model(model_ids[0])
        
        # get the metalearner model
        meta = h2o.get_model(m.metalearner()['name'])
        
        # get varimp_df from metalearner
        if ('glm' in meta.algo) is True:
            varimp_df = pd.DataFrame.from_dict((meta.coef()), orient = 'index')
            varimp_df = varimp_df[1:] # omit Intercept
        else:
            varimp_df = pd.DataFrame(meta.varimp())
        
        model_list = []
        
        for model in m.params['base_models']['actual']:
            model_list.append(model['name'])
        
        print(model_list)
        
        # create two dictionaries for storing variable importance and rmse
        var_imp_models = dict([(key, []) for key in model_list])
        rmse_df = dict([(key, []) for key in model_list])
        
        
        # get variable importance and rmse from base learners
        for model in model_list:
            tmp_model = h2o.get_model(str(model))
            
            # check if tmp_model has varimp()
            if tmp_model.varimp() is None:
                print(str(model))
                del var_imp_models[str(model)]
            else:
                # check if tmp_model is glm - it has no varimp() but coef()
                if ('glm' in tmp_model.algo) is True:
                    tmp_var_imp = pd.DataFrame.from_dict(tmp_model.coef(), orient = 'index').rename(columns={0:'scaled_importance'})
                    tmp_var_imp = tmp_var_imp[1:] # omit Intercept
                    tmp_var_imp.insert(loc = 0, column = 'variable', value = tmp_var_imp.index) # reset index of rows into column
                else:
                    tmp_var_imp = tmp_model.varimp(use_pandas=True).iloc[:,[0,2]]
                
                tmp_rmse = tmp_model.rmse()
                var_imp_models[str(model)].append(tmp_var_imp)
                rmse_df[str(model)].append(tmp_rmse)
                
        if original_scale is False:
            rmse_df = pd.DataFrame(rmse_df.values())
            rmse_sum = rmse_df.sum()[0]
            rmse_scale = rmse_sum / rmse_df
            
            x = rmse_scale.values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            rmse_scale = pd.DataFrame(x_scaled)
            rmse_scale = pd.DataFrame(data=rmse_scale.values,index=model_list)
            
            for idx in rmse_scale.iterrows():
                var_imp_models[str(idx[0])][0]['scaled_importance'] = var_imp_models[str(idx[0])][0].values[0:,1] * idx[1].values
                
        elif original_scale is True:
            meta_scale = varimp_df
            for idx in meta_scale.iterrows():
                if ('glm' in meta.algo) is True:
                    var_imp_models[str(idx[0])][0]['scaled_importance'] = var_imp_models[str(idx[0])][0].values[0:,1] * float(idx[1])
                else:
                    var_imp_models[str(idx[1][0])][0]['scaled_importance'] = var_imp_models[str(idx[1][0])][0]['scaled_importance'] * idx[1][3]
            
        # new dataframe init     
        scaled_var_imp_df = pd.DataFrame()
        
        for idx in var_imp_models.keys():
            df_tmp = var_imp_models[str(idx)][0]['scaled_importance']
            df_tmp.index = var_imp_models[str(idx)][0]['variable']
            scaled_var_imp_df = pd.concat([scaled_var_imp_df, df_tmp], axis =1, sort = False)
        
        # sum rows by index, NaNs are consdered as zeros
        #Total sum per row: 
        scaled_var_imp_df.loc[:,'Total'] = scaled_var_imp_df.sum(axis=1)
        
        # Drop variables by a fs_threshold condition
        scaled_var_imp_df = scaled_var_imp_df[scaled_var_imp_df.Total > fs_threshold]
        
        # Sort by 'Total' values
        scaled_var_imp_df_sorted = scaled_var_imp_df.sort_values(by=['Total'], ascending = False)
        
        # Plot and save bar chart
        plt.rcParams['xtick.labelsize'] = 4
        ax = scaled_var_imp_df_sorted.plot.bar(y='Total', rot=90)
        plt.tight_layout()
        plt.savefig('FS_result_h2o.pdf', format='pdf', dpi=1200)

# --------------------------------------------------------------


# ----------------------------------------------------
# 2nd option - use_classic_approach
# ----------------------------------------------------

if use_classic_approach is True and perform_FS is False:

    # checking if my_10cv_FS_dir or my_10cv_orig_dir has train/test 10cv pairs
    print('\n' + 'Checking for non-empty dir (' + str(my_10cv_FS_dir) + ')' + '\n')

# General if-else statement to check dir if it contains t-* files
    if len(os.listdir(my_10cv_FS_dir)) > 0:
        for file_idx in os.listdir(my_10cv_FS_dir):
            m = bool(re.match(r"t-[0-9][A-Z]*", file_idx))
            if m is True:
                print('File ' + str(file_idx) + ' will be used for creating 10cv')
    else:
        print('t-*.txt files were not found in' + str(my_10cv_FS_dir) + '. Using original data and making n-fold cv ...')

        # divide between X(input) and y(output)
        # First column contains group indicies
        # Last column contains output
        ncols = data.shape[1] - 1
        nrows = data.shape[0]

        X = data.drop(data.columns[[0, ncols]], axis=1)
        y = data[data.columns[ncols]]

        # needed to make cv by groups - first column contains indicies!
        groups = data[data.columns[[index_column - 1]]]

        # Define how many fold there will be
        gkf = GroupKFold(n_splits=no_folds)
        cv_fold = 0

        for train_index, test_index in gkf.split(X, y, groups=groups):
            cv_fold += 1
            print("CV fold: ", cv_fold)
            print("Train Index: ", train_index)
            print("Test Index: ", test_index, "\n")

            trainX_data = X.loc[train_index]
            trainy_data = y.loc[train_index]

            testX_data = X.loc[test_index]
            testy_data = y.loc[test_index]

            # Save original 10cv folds with all features
            train_set = pd.concat([trainX_data, trainy_data], axis=1)
            test_set = pd.concat([testX_data, testy_data], axis=1)

            # generate a file name based on the id and record and save orig 10cv datasets
            file_name_train = "10cv_orig_" + str(core_filename) + "_no" + str(cv_fold) + ".txt"
            file_name_test = "t-10cv_orig_" + str(core_filename) + "_no" + str(cv_fold) + ".txt"

            train_set.to_csv(r'./10cv_orig/' + file_name_train, index=False, sep="\t")
            test_set.to_csv(r'./10cv_orig/' + file_name_test, index=False, sep="\t")

            # functionality to manually add features, eg. 'Time_min' in dissolution profiles
            if len(include_features) > 0:
                include_features_df_train = X.loc[train_index]
                include_features_df_test = X.loc[test_index]
                include_features_df_train = include_features_df_train[include_features]
                include_features_df_test = include_features_df_test[include_features]

                trainX_data = pd.concat([include_features_df_train, trainX_data], axis=1)
                testX_data = pd.concat([include_features_df_test, testX_data], axis=1)
                trainX_data = trainX_data.loc[:, ~trainX_data.columns.duplicated()]
                testX_data = testX_data.loc[:, ~testX_data.columns.duplicated()]

            train_set = pd.concat([trainX_data, trainy_data], axis=1)
            test_set = pd.concat([testX_data, testy_data], axis=1)

            ncols = train_set.shape[1] - 1
            nrows = train_set.shape[0]

            print('nrows for' + aml2_name + ' project train dataset = ', nrows)
            print('ncols for train dataset = ', ncols)

            # save datasets
            file_name_train = "10cv_" + str(core_filename) + str(ncols) + "_in" + "_no" + str(
                cv_fold) + ".txt"
            file_name_test = "t-10cv_" + str(core_filename) + str(ncols) + "_in" + "_no" + str(
                cv_fold) + ".txt"

            train_set.to_csv(r'./10cv_FS/' + file_name_train, index=False, sep="\t")
            test_set.to_csv(r'./10cv_FS/' + file_name_test, index=False, sep="\t")
        # split loop end


    # Load testing data in a loop and make folds based on them
    # 1) List all files with pattern 't-*.txt' in ./10cv_FS
    all_filenames = [i for i in glob.glob('./10cv_FS/' + 't-10cv_' + '*')]
    
    # 2) Sort list of filenames from 1 to 10
    all_filenames.sort(key = lambda x: int(x.split('_no')[1].split('.')[0]))
    
    # 3) read all files in a list into a data_frame and make indicies for each t-file
    df_classic = pd.DataFrame()
    df_classic = pd.concat([pd.read_csv(all_filenames[index], header = [0], sep = '\t', engine = 'python').assign(Fold_no=index+1) for index in range(len(all_filenames))])
    
    # index of the output column
    y_idx = df_classic.columns[df_classic.shape[1]-2]
    training_frame = h2o.H2OFrame(df_classic)
    
    # assign fold column name
    assignment_type = 'Fold_no'
    
    # set new AutoML options
    aml_10cv = H2OAutoML(max_runtime_secs = h2o_max_runtime_secs_2nd_time_10cv,
                         seed = my_seed_classic_approach,
                         project_name = aml2_name,
                         nfolds = no_folds,
                         export_checkpoints_dir = str(my_export_dir),
                         keep_cross_validation_predictions = my_keep_cross_validation_predictions,
                         keep_cross_validation_models = my_keep_cross_validation_models,
                         keep_cross_validation_fold_assignment = my_keep_cross_validation_fold_assignment,
                         verbosity = 'info',
                         sort_metric = 'RMSE')
    
    # train AutoML with fold_column!
    aml_10cv.train(y = y_idx, training_frame = training_frame, fold_column = assignment_type)
    
    # save h2o model
    print('Saving leader h2o model in ./model_10cv and ./test_external')
    my_10cv_model_path = h2o.save_model(aml_10cv.leader, path = './model_10cv')
    
    h2o.save_model(aml_10cv.leader, path = './test_external')
    
    print('')
    print('The final model afer k-fold cv is located at: ')
    print(str(my_10cv_model_path))
    print('')
    
    # Download POJO or MOJO
    if save_pojo_or_mojo is True:
        if aml_10cv.leader.have_pojo is True:
            aml_10cv.leader.download_pojo(get_genmodel_jar = True, path = './pojo_or_mojo_10cv')
        if aml_10cv.leader.have_mojo is True:
            aml_10cv.leader.download_mojo(get_genmodel_jar = True, path = './pojo_or_mojo_10cv')
    
    # get the best model key
    model_key = aml_10cv.leader.key
    
    # get the models id
    model_ids = list(aml_10cv.leaderboard['model_id'].as_data_frame().iloc[:,0])
    
    # get the best model
    m = h2o.get_model(aml_10cv.leader.key)
    print('Leader model: ')
    print(m.key)
    
    
    if ("StackedEnsemble" in aml_10cv.leader.key) is True:
        # get the metalearner name
        se_meta_model = h2o.get_model(m.metalearner()['name'])
        my_se_meta_model_path = h2o.save_model(se_meta_model, path = './model_10cv')
        print('')
        print('The meta model of the best model is located at: ')
        print(str(my_se_meta_model_path))
        print('')
        
        h2o_cv_data = se_meta_model.cross_validation_holdout_predictions()
        pred_obs = h2o_cv_data.cbind([training_frame[training_frame.col_names[len(training_frame.col_names)-2]], training_frame['Fold_no']])
            
        # get a list of models - save and print out
        model_list = []
        print('Saving constituents of the StackedEnsemble')
        for model in m.params['base_models']['actual']:
            model_list.append(model['name'])
            my_tmp_model_path = h2o.save_model(h2o.get_model(str(model['name'])), path = './model_10cv')
            print(str(my_tmp_model_path))
            
        print('Stacked Ensemble model contains: ')
        print(model_list)
        
        
    else:
        h2o_cv_data = m.cross_validation_holdout_predictions()
        pred_obs = h2o_cv_data.cbind([training_frame[training_frame.col_names[len(training_frame.col_names)-2]],training_frame['Fold_no']])
        
# ------------------------------------------------------------------------------


# ----------------------------------------------------------------
# 3rd option - use_classic_approach is False and perfor_FS is True
# ----------------------------------------------------------------

if use_classic_approach is False and perform_FS is True:
        # Perform k-fold cv 
        # split on train - test dataset by group - according to no_folds
        gkf = GroupKFold(n_splits=no_folds)
        cv_fold = 0
        
        for train_index, test_index in gkf.split(X, y, groups=groups):
            cv_fold += 1
            print("CV fold: ", cv_fold)
            print("Train Index: ", train_index)
            print("Test Index: ", test_index, "\n")
            
            trainX_data = X.loc[train_index]
            trainy_data = y.loc[train_index]
            
            testX_data = X.loc[test_index]
            testy_data = y.loc[test_index]
            
            # Save original 10cv folds with all features
            train_set = pd.concat([trainX_data, trainy_data], axis=1)
            test_set = pd.concat([testX_data, testy_data], axis=1)
            
            # generate a file name based on the id and record and save orig 10cv datasets
            file_name_train = "10cv_orig_" + str(core_filename) + "_no" + str(cv_fold) + ".txt"
            file_name_test = "t-10cv_orig_" + str(core_filename) + "_no" + str(cv_fold) + ".txt"
            
            train_set.to_csv(r'./10cv_orig/' + file_name_train, index=False, sep="\t")
            test_set.to_csv(r'./10cv_orig/' + file_name_test, index=False, sep="\t")
            print(model_key)
            if ('StackedEnsemble' in model_key) is True:
                 # Remove features that score below threshold
                 trainX_data = trainX_data[scaled_var_imp_df.index.tolist()]
                 # trainy_data stays the same
                 testX_data = testX_data[scaled_var_imp_df.index.tolist()]
            elif ('StackedEnsemble' in model_key) is False:
                # Remove features that score below threshold
                trainX_data = trainX_data[scaled_var_imp_df['variable']]
                # trainy_data stays the same
                testX_data = testX_data[scaled_var_imp_df['variable']]
                # testy_data stays the same
            
            # functionality to manually add features, eg. 'Time_min' in dissolution profiles
            if len(include_features) > 0:
                include_features_df_train = X.loc[train_index]
                include_features_df_test = X.loc[test_index]
                include_features_df_train = include_features_df_train[include_features]
                include_features_df_test = include_features_df_test[include_features]
                
                trainX_data = pd.concat([include_features_df_train, trainX_data], axis = 1)
                testX_data = pd.concat([include_features_df_test, testX_data], axis = 1)
                trainX_data = trainX_data.loc[:,~trainX_data.columns.duplicated()]
                testX_data = testX_data.loc[:,~testX_data.columns.duplicated()]
            
            train_set = pd.concat([trainX_data, trainy_data], axis=1)
            test_set = pd.concat([testX_data, testy_data], axis=1)
            
            ncols=train_set.shape[1]-1
            nrows=train_set.shape[0]
            
            print('nrows for' + aml2_name + ' project train dataset = ',nrows)
            print('ncols for train dataset = ',ncols)
            
            # save datasets after feature selection
            file_name_train = "10cv_" + str(core_filename)+ "_FS_to_" + str(ncols) + "_in" +"_no"+str(cv_fold)+".txt"
            file_name_test = "t-10cv_"+str(core_filename)+ "_FS_to_" + str(ncols) + "_in" +"_no"+str(cv_fold)+".txt"
            
            train_set.to_csv(r'./10cv_FS/' + file_name_train, index=False, sep="\t")
            test_set.to_csv(r'./10cv_FS/' + file_name_test, index = False, sep="\t")
        # split loop end
            
        # Load testing data in a loop and make folds based on them 
        # 1) List all files with pattern 't-*.txt' in ./10cv_orig
        all_filenames = [i for i in glob.glob('./10cv_FS/t-*.txt')]
        # 2) Sort list of filenames from 1 to 10
        all_filenames.sort(key = lambda x: int(x.split('_no')[1].split('.')[0]))
        # 3) read all files in a list into a data_frame and make indicies for each t-file
        df_new_approach = pd.concat([pd.read_csv(all_filenames[index], header = [0], sep = '\t', engine = 'python').assign(Fold_no=index+1) for index in range(len(all_filenames))])
            
        # index of the output column
        y_idx = df_new_approach.columns[df_new_approach.shape[1]-2]
        training_frame = h2o.H2OFrame(df_new_approach)
        
        # assign fold column name
        assignment_type = 'Fold_no'
        
        # set new AutoML options
        aml_10cv = H2OAutoML(max_runtime_secs = h2o_max_runtime_secs_2nd_time_10cv,
                             seed = my_seed_FS_10cv,
                             project_name = aml2_name,
                             nfolds = no_folds,
                             export_checkpoints_dir = str(my_export_dir),
                             keep_cross_validation_predictions = my_keep_cross_validation_predictions,
                             keep_cross_validation_models = my_keep_cross_validation_models,
                             keep_cross_validation_fold_assignment = my_keep_cross_validation_fold_assignment,
                             verbosity = 'info',
                             sort_metric = 'RMSE')
        
        # train AutoML with fold_column!
        aml_10cv.train(y = y_idx, training_frame = training_frame, fold_column = assignment_type)
        
        # save h2o model
        print('Saving leader h2o model in ./model_10cv and ./test_external')
        my_10cv_model_path = h2o.save_model(aml_10cv.leader, path = './model_10cv')
        
        print('')
        print('The final model afer k-fold cv is located at: ')
        print(str(my_10cv_model_path))
        print('')
        
        h2o.save_model(aml_10cv.leader, path = './test_external')
        
        # Download POJO or MOJO
        if save_pojo_or_mojo is True:
            if aml_10cv.leader.have_pojo is True:
                aml_10cv.leader.download_pojo(get_genmodel_jar = True, path = './pojo_or_mojo_10cv')
            if aml_10cv.leader.have_mojo is True:
                aml_10cv.leader.download_mojo(get_genmodel_jar = True, path = './pojo_or_mojo_10cv')
        
        
        # get the models id
        model_ids = list(aml_10cv.leaderboard['model_id'].as_data_frame().iloc[:,0])
        # get the best model
        m = h2o.get_model(aml_10cv.leader.key)
        print('Leader model: ')
        print(m.key)
        
        
        if ("StackedEnsemble" in aml_10cv.leader.key) is True:
            # get the metalearner name
            se_meta_model = h2o.get_model(m.metalearner()['name'])
            
            my_se_meta_model_path = h2o.save_model(se_meta_model, path = './model_10cv')
            print('')
            print('The meta model of the best model is located at: ')
            print(str(my_se_meta_model_path))
            print('')
            
            h2o_cv_data = se_meta_model.cross_validation_holdout_predictions()
            pred_obs = h2o_cv_data.cbind([training_frame[training_frame.col_names[len(training_frame.col_names)-2]],training_frame['Fold_no']])
            
            # get a list of models - save and print out
            model_list = []
            
            print('Saving constituents of the StackedEnsemble')
            for model in m.params['base_models']['actual']:
                model_list.append(model['name'])
                my_tmp_model_path = h2o.save_model(h2o.get_model(str(model['name'])), path = './model_10cv')
                print(str(my_tmp_model_path))
            
            print('Stacked Ensemble model contains: ')
            print(model_list)
            
            
        else:
            h2o_cv_data = m.cross_validation_holdout_predictions()
            pred_obs = h2o_cv_data.cbind([training_frame[training_frame.col_names[len(training_frame.col_names)-2]], training_frame['Fold_no']])


# ------------------------------------------------------------------------------------------


# ----------------------------------------------
# 
# report in our 10cv manner
# 
# ----------------------------------------------

# Store results in:
pred_obs = pd.DataFrame(h2o.as_list(pred_obs))

# define function to calculate R2 and RMSE based on sklearn methods r2_score() and mean_squared_error()
def r2_rmse(g):
    r2 = r2_score(g.iloc[:,[1]], g.iloc[:,[0]]) # r2_score is giving negative values - check/test with validation
    rmse = np.sqrt(mean_squared_error(g.iloc[:,[1]], g.iloc[:,[0]]))
    return pd.Series(dict(r2 = r2, rmse = rmse))

# get training and testing file names
if perform_FS is True:
    testing_filenames = [i for i in glob.glob('./10cv_FS/t-10cv*')]
    traning_filenames = [i for i in glob.glob('./10cv_FS/10cv*')]
    
elif use_classic_approach is True:
    testing_filenames = [i for i in glob.glob('./10cv_FS/' + 't-' + '*')]
    traning_filenames = [i for i in glob.glob('./10cv_FS/' + 't-' + '*')]

# sort training and testing files
testing_filenames.sort(key = lambda x: int(x.split('_no')[1].split('.')[0]))
traning_filenames.sort(key = lambda x: int(x.split('_no')[1].split('.')[0]))

# Calculate RMSE for each fold grouped by value in column 'Fold_no'
# and store values in r2_rmse_table
# -----------------------------------------------------------------
r2_rmse_table = pd.DataFrame(pred_obs.groupby('Fold_no').apply(r2_rmse).reset_index())

# write data to RESULTS.txt file

report_file = 'RESULTS.txt'
fd = open(report_file, 'w')
fd.write('Report file'+'\n')
       
for row_index in range(len(r2_rmse_table.index)):
    fd.write('Iteration '+ str(int(r2_rmse_table.iloc[row_index][0]))+'\n')
    fd.write('Training file '+ str(traning_filenames[row_index])+'\n')
    fd.write('Test file '+ str(testing_filenames[row_index])+'\n')
    fd.write('RMSE = '+str(r2_rmse_table.iloc[row_index][2])+'\n')
    fd.write('R2 = '+str(r2_rmse_table.iloc[row_index][1])+'\n')
    tmp_pred_obs = pred_obs.loc[pred_obs['Fold_no'] == (row_index + 1)]
    tmp_pred_obs.to_csv(fd,sep='\t')
    fd.write(''+'\n')

fd.close()

average_RMSE = statistics.mean(r2_rmse_table.iloc[:]['rmse'])
average_R2 = statistics.mean(r2_rmse_table.iloc[:]['r2'])

print('errors ',r2_rmse_table)
print('average_RMSE = ',average_RMSE)
print('average_R2 = ',average_R2)
print('overall_results')
print(pred_obs)
overall_RMSE = np.sqrt(mean_squared_error(pred_obs.iloc[:,[1]], pred_obs.iloc[:,[0]]))
overall_R2=r2_score(pred_obs.iloc[:,[1]], pred_obs.iloc[:,[0]])

fd = open(report_file,'a')
fd.write('Average RMSE = '+'\t'+str(average_RMSE) + '\n')
fd.write('----------------------' + '\n')
fd.write('Average R2 = '+str(average_R2) + '\n')
fd.write('----------------------' + '\n')
fd.write('' + '\n')
fd.write('----------------------' + '\n')
fd.write('Overall data' + '\n')
fd.write('----------------------' + '\n')
fd.write('Overall RMSE = '+str(overall_RMSE) + '\n')
fd.write('Overall R2 = '+str(overall_R2) + '\n')
fd.write('------------------------' + '\n')
pred_obs.to_csv(fd,sep='\t')
fd.write('------------------------' + '\n')
fd.write('END OF FILE' + '\n')
fd.close()

# Write plain RMSE in 'short_outfile.txt'
report_file_short='short_outfile.txt'

fd = open(report_file_short, 'w')
fd.write(str(overall_RMSE))
fd.close()
