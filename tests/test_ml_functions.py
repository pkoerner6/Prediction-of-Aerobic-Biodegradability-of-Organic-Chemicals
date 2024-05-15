# import sys
# import os
# import numpy as np
# import pandas as pd
# from xgboost import XGBClassifier

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from code_files.processing_functions import get_speciation_col_names
# from code_files.processing_functions import load_class_data_paper

# from code_files.ml_functions import get_class_results
# from code_files.ml_functions import print_class_results
# from code_files.ml_functions import split_classification_df_with_fixed_test_set
# from code_files.ml_functions import skf_class_fixed_testset
# from code_files.ml_functions import get_balanced_data_adasyn
# from code_files.ml_functions import run_balancing_and_training
# from code_files.ml_functions import skf_classification
# from code_files.ml_functions import split_regression_df_with_grouping
# from code_files.ml_functions import split_regression_df_with_grouping_and_fixed_test_set
# from code_files.ml_functions import create_train_test_sets_regression
# from code_files.ml_functions import kf_regression
# from code_files.ml_functions import report_perf_hyperparameter_tuning
# from code_files.ml_functions import get_Huang_Zhang_regression_parameters
# from code_files.ml_functions import train_XGBRegressor_Huang_Zhang
# from code_files.ml_functions import train_XGBClassifier
# from code_files.ml_functions import train_XGBClassifier_on_all_data
# from code_files.ml_functions import plot_regression_error
# from code_files.ml_functions import analyze_regression_results_and_plot


# def test_get_class_results():
#     y_test_dict = {"y_true": {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}}
#     y_test = pd.DataFrame.from_dict(y_test_dict)
#     prediction = np.array([0, 0, 1, 0, 1, 1], np.int32)
#     accuracy, f1, sensitivity, specificity = get_class_results(true=y_test.to_numpy(), pred=prediction)
#     assert accuracy == 0.5
#     assert f1 == 0.4
#     assert sensitivity == 0.5
#     assert specificity == 0.5


# def test_split_classification_df_with_fixed_test_set(class_df_long, class_curated):
#     nsplits = 3
#     df_test = class_df_long.copy()[:10]
#     cols = ["cas", "smiles", "y_true"]
#     train_sets1, test_sets1 = split_classification_df_with_fixed_test_set(
#         df=class_df_long,
#         df_test=df_test,
#         nsplits=nsplits,
#         random_seed=42,
#         cols=cols,
#         paper=False,
#     )
#     assert len(train_sets1) == nsplits
#     assert len(test_sets1) == nsplits
#     for test in test_sets1:
#         for cas in test["cas"]:
#             for i in range(nsplits):
#                 assert cas not in train_sets1[i]["cas"]
#     train_sets2, test_sets2 = split_classification_df_with_fixed_test_set(
#         df=class_curated,
#         df_test=df_test,
#         nsplits=nsplits,
#         random_seed=42,
#         cols=cols,
#         paper=False,
#     )
#     for i in range(nsplits):
#         assert test_sets1[i]["cas"].to_list() == test_sets2[i]["cas"].to_list()
#     for test in test_sets2:
#         for cas in test["cas"]:
#             for i in range(nsplits):
#                 assert cas not in train_sets2[i]["cas"]


# def test_skf_class_fixed_testset(class_df_long):
#     nsplits = 3
#     df_test = class_df_long.copy()[:10]
#     cols = ["cas", "smiles", "y_true"]

#     (
#         x_train_fold_lst,
#         y_train_fold_lst,
#         x_test_fold_lst,
#         y_test_fold_lst,
#         df_test_lst,
#         test_set_sizes,
#     ) = skf_class_fixed_testset(
#         df=class_df_long,
#         df_test=df_test,
#         nsplits=nsplits,
#         random_seed=42,
#         include_speciation=False,
#         cols=cols,
#         target_col="y_true",
#         paper=False,
#     )
#     lsts = [x_train_fold_lst, y_train_fold_lst, x_test_fold_lst, y_test_fold_lst, df_test_lst, test_set_sizes]
#     for lst in lsts:
#         assert len(lst) == nsplits
#     assert (cols + ["inchi_from_smiles"]) == list(df_test_lst[0].columns)
#     for i in range(len(y_test_fold_lst)):
#         assert df_test_lst[i]["y_true"].to_list() == y_test_fold_lst[i].to_list()

#     _, _, x_test_fold_lst2, y_test_fold_lst2, _, _ = skf_class_fixed_testset(
#         df=df_test,
#         df_test=df_test,
#         nsplits=nsplits,
#         random_seed=42,
#         include_speciation=False,
#         cols=cols,
#         target_col="y_true",
#         paper=False,
#     )
#     for i in range(len(x_test_fold_lst)):
#         assert x_test_fold_lst[i].tolist() == x_test_fold_lst2[i].tolist()
#         assert y_test_fold_lst[i].tolist() == y_test_fold_lst2[i].tolist()

#     cols += get_speciation_col_names()
#     _, _, _, _, df_test_lst3, _ = skf_class_fixed_testset(
#         df=class_df_long,
#         df_test=df_test,
#         nsplits=nsplits,
#         random_seed=42,
#         include_speciation=True,
#         cols=cols,
#         target_col="y_true",
#         paper=False,
#     )
#     assert (cols + ["inchi_from_smiles"]) == list(df_test_lst3[0].columns)


# def test_get_balanced_data_adasyn(class_df_long):
#     nsplits = 3
#     cols = ["cas", "smiles", "y_true"]
#     _, _, class_df = load_class_data_paper()
#     df_test = class_df_long.copy()[:10]
#     x_train_fold_lst, y_train_fold_lst, _, _, _, _ = skf_class_fixed_testset( 
#         df=class_df,
#         df_test=df_test,
#         nsplits=nsplits,
#         random_seed=42,
#         include_speciation=False,
#         cols=cols,
#         target_col="y_true",
#         paper=False,

#     )
#     for x_train, y_train in zip(x_train_fold_lst, y_train_fold_lst):
#         class_1 = len(y_train[y_train == 1])
#         class_0 = len(y_train[y_train == 0])
#         x_train_fold, y_train_fold = get_balanced_data_adasyn(random_seed=42, x=x_train, y=y_train)
#         entries_class_1 = len(y_train_fold[y_train_fold == 1])
#         entries_class_0 = len(y_train_fold[y_train_fold == 0])
#         ratio = entries_class_0 / entries_class_1
#         assert (ratio < 1.1) or (ratio > 0.9)
#         assert class_0 == entries_class_0  # only upsample smaller class, in this case 1


# def test_split_regression_df_with_grouping(regression_paper_full):
#     nsplits = 3
#     train_dfs, test_dfs = split_regression_df_with_grouping(
#         df=regression_paper_full,
#         nsplits=nsplits,
#         column_for_grouping="cas",
#         random_seed=42,
#     )
#     assert len(train_dfs) == nsplits
#     assert len(test_dfs) == nsplits
#     for train_df, test_df in zip(train_dfs, test_dfs):
#         for cas in train_df.cas:
#             assert cas not in test_df.cas

#     train_dfs, test_dfs = split_regression_df_with_grouping(
#         df=regression_paper_full,
#         nsplits=nsplits,
#         column_for_grouping="cas",
#         random_seed=42,
#     )
#     for col in get_speciation_col_names():
#         assert col in train_dfs[0].columns
#         assert col in test_dfs[0].columns


# def test_split_regression_df_with_grouping_and_fixed_test_set(regression_paper_full):
#     nsplits = 3
#     df_test = regression_paper_full[:5]
#     train_dfs1, test_dfs1 = split_regression_df_with_grouping_and_fixed_test_set(
#         df=regression_paper_full,
#         df_test=df_test,
#         nsplits=nsplits,
#         column_for_grouping="cas",
#         random_seed=42,
#     )
#     train_dfs2, test_dfs2 = split_regression_df_with_grouping_and_fixed_test_set(
#         df=df_test,
#         df_test=df_test,
#         nsplits=nsplits,
#         column_for_grouping="cas",
#         random_seed=42,
#     )
#     assert len(train_dfs1) == nsplits
#     assert len(test_dfs1) == nsplits
#     assert len(train_dfs2) == nsplits
#     assert len(test_dfs2) == nsplits
#     for train_df1, test_df1, train_df2, test_df2 in zip(train_dfs1, test_dfs1, train_dfs2, test_dfs2):
#         assert list(test_df1.cas) == list(test_df2.cas)
#         for cas1, cas2 in zip(test_df1.cas, test_df2.cas):
#             assert cas1 not in train_df1.cas
#             assert cas2 not in train_df2.cas


# def test_create_train_test_sets_regression(regression_paper_full):
#     nsplits = 3
#     train_dfs, test_dfs = split_regression_df_with_grouping(
#         df=regression_paper_full,
#         nsplits=nsplits,
#         column_for_grouping="cas",
#         random_seed=42,
#     )
#     x_train, y_train, x_test, y_test = create_train_test_sets_regression(
#         train_df=train_dfs[0], test_df=test_dfs[0], include_speciation=False
#     )
#     assert list(y_train) == list(train_dfs[0]["biodegradation_percent"])
#     assert list(y_test) == list(test_dfs[0]["biodegradation_percent"])
#     assert len(x_train[0]) == 172
#     assert type(x_train) == np.ndarray
#     assert type(x_train[0]) == np.ndarray
#     assert len(x_test[0]) == 172
#     assert type(x_test) == np.ndarray
#     assert type(x_test[0]) == np.ndarray
#     assert len(x_train) == len(y_train)
#     assert len(x_test) == len(y_test)


# def test_train_XGBClassifier(class_df_long):
#     nsplits = 3
#     accu, f1, sensitivity, specificity = train_XGBClassifier(
#         df=class_df_long,
#         random_seed=42,
#         nsplits=nsplits,
#         use_adasyn=True,
#         include_speciation=False,
#         df_test=class_df_long,
#         dataset_name="classification_paper_full_test",
#         target_col="y_true",
#     )
#     assert len(accu) == nsplits
#     assert len(f1) == nsplits
#     assert len(sensitivity) == nsplits
#     assert len(specificity) == nsplits


# def test_train_XGBClassifier_on_all_data(class_df_long):
#     model_class = train_XGBClassifier_on_all_data(
#         df=class_df_long, random_seed=42, include_speciation=False
#     )
#     assert model_class.__class__.__name__ == "XGBClassifier"
