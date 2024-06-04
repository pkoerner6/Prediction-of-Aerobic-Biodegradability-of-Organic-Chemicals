
import numpy as np
import pandas as pd
import structlog
import sys
import os
import statistics
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

log = structlog.get_logger()
from typing import List, Dict, Tuple


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_files.processing_functions import load_class_data_paper


######################################################
# Adapted from pyADA https://github.com/jeffrichardchemistry/pyADA/blob/main/pyADA/pyADA.py

from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Smetrics:
    def __init__(self):
        self.maxdata = None
        self.mindata = None 


class Similarity:
    """
        All similarity calculations have a range of [0, 1]
    """
    def __init__(self):        
        pass    
    
    def __coefs(self, vector1, vector2):
        A = np.array(vector1).astype(int)
        B = np.array(vector2).astype(int)

        AnB = A & B #intersection
        onlyA = np.array(B) < np.array(A) #A is a subset of B
        onlyB = np.array(A) < np.array(B) #B is a subset of A
        return AnB,onlyA,onlyB
    

    def tanimoto_similarity(self, vector1, vector2):
        """
        Structural similarity calculation based on tanimoto index. T(A,B) = (A ^ B)/(A + B - A^B)
        """
        AnB, onlyA, onlyB = Similarity.__coefs(self, vector1=vector1, vector2=vector2)
        return AnB.sum() / (onlyA.sum() + onlyB.sum() + AnB.sum())


class ApplicabilityDomain:
    def __init__(self, verbose=False):
        self.__sims = Similarity()    
        self.__verbose = verbose
        self.similarities_table_ = None
                
    def analyze_similarity(self, base_test, base_train, similarity_metric='tanimoto'):
        """
        Analysis of the similarity between molecular fingerprints
        using different metrics. A table (dataframe pandas) will be
        generated with the coefficients: Average, median, std,
        maximum similarity and minimum similarity, for all compounds
        in the test database in relation to the training database.
        The alpha and beta parameters are only for 'tversky' metric.
        """

        similarities = {}

        # get dictionary of all data tests similarities
        def get_dict(base_train, i_test, similarities, n):
            get_tests_similarities = [0]*len(base_train)
            for i, i_train in enumerate(base_train):
                if similarity_metric == 'tanimoto':
                    get_tests_similarities[i] = (self.__sims.tanimoto_similarity(i_test, i_train))               
                else:
                    log.error("This similarity_metric does not exist")
            similarities['Sample_test_{}'.format(n)] = np.array(get_tests_similarities)
            return similarities
        
        if self.__verbose:
            with tqdm(total=len(base_test)) as progbar:
                for n,i_test in enumerate(base_test):
                    similarities = get_dict(base_train, i_test, similarities, n)
                    progbar.update(1)
        else:
            for n,i_test in enumerate(base_test):            
                similarities = get_dict(base_train, i_test, similarities, n)
                    
        self.similarities_table_ = pd.DataFrame(similarities)
        
        analyze = pd.concat([self.similarities_table_.mean(),
                             self.similarities_table_.median(),
                             self.similarities_table_.std(),
                             self.similarities_table_.max(),
                             self.similarities_table_.min()],
                             axis=1)        
        analyze.columns = ['Mean', 'Median', 'Std', 'Max', 'Min']
        
        return analyze
            
    
    def fit(self, model, base_test, base_train, y_true, isTensorflow=False,
            threshold_reference = 'max', threshold_step = (0, 1, 0.05),
            similarity_metric='tanimoto', alpha = 1, beta = 1, metric_evaliation='rmse'):
        
        #reference parameters
        if threshold_reference.lower() == 'max':
            thref = 'Max'
        elif threshold_reference.lower() == 'average':
            thref = 'Mean'
        elif threshold_reference.lower() == 'std':
            thref = 'Std'
        elif threshold_reference.lower() == 'median':
            thref = 'Median'
        else:
            thref = 'Max'
        
        #Get analysis table
        table_analysis = ApplicabilityDomain.analyze_similarity(self, base_test=base_test, base_train=base_train,
                                                            similarity_metric=similarity_metric)
                                                            # alpha=alpha, beta=beta) # TODO
        table_analysis.index = np.arange(0, len(table_analysis), 1)
        
        results = {}
        total_thresholds = np.arange(threshold_step[0], threshold_step[1], threshold_step[2])
        
        def get_table(thresholds, table_analysis, thref, samples_GT_threshold, base_test, isTensorflow, model, y_true, metric_evaliation, results):
            samples_LT_threshold = table_analysis.loc[table_analysis[thref] < thresholds] #get just samples < threshold
            new_xitest = base_test[samples_GT_threshold.index, :] #get samples > threshold in complete base_test
            if isTensorflow:
                new_ypred = model.predict(new_xitest) #precit y_pred
                new_ypred[new_ypred <= 0.5] = 0
                new_ypred[new_ypred > 0.5] = 1
                new_ypred = new_ypred.astype(int)
            else:
                new_ypred = model.predict(new_xitest) #precit y_pred
            new_ytrue = y_true[samples_GT_threshold.index] #get y_true (same index of xi_test) (y_true must be a array 1D in this case)
            
            #calc of ERROR METRICS (EX: RMSE) or correlation methods
            if metric_evaliation == 'acc':
                error_ = accuracy_score(y_true=new_ytrue, y_pred=new_ypred)
            else:
                log.error("This metric_evaliation is not defined")
                
            results['Threshold {}'.format(thresholds.round(5))] = [[error_],np.array(samples_LT_threshold.index)]
            return results


        if self.__verbose:
            
            for thresholds in tqdm(total_thresholds):
                samples_GT_threshold = table_analysis.loc[table_analysis[thref] >= thresholds] #get just samples > threshold
                if len(samples_GT_threshold) == 0:
                    print('\nStopping with Threshold {}. All similarities are less than or equal {} '.format(thresholds, thresholds))
                    break
                results = get_table(thresholds, table_analysis, thref, samples_GT_threshold, base_test, isTensorflow, model, y_true, metric_evaliation, results)
                
            return results
        
        else:            
            for thresholds in total_thresholds:
                samples_GT_threshold = table_analysis.loc[table_analysis[thref] >= thresholds] #get just samples > threshold
                if len(samples_GT_threshold) == 0:
                    print('\nStopping with Threshold {}. All similarities are less than or equal {} '.format(thresholds, thresholds))
                    break
                results = get_table(thresholds, table_analysis, thref, samples_GT_threshold, base_test, isTensorflow, model, y_true, metric_evaliation, results)
                
            return results


######################################################


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_curated_scs = pd.read_csv(
        "datasets/curated_data/class_curated_scs.csv", index_col=0
    )
    df_curated_biowin = pd.read_csv(
        "datasets/curated_data/class_curated_biowin.csv", index_col=0
    )
    df_curated_final = pd.read_csv(
        "datasets/curated_data/class_curated_final.csv", index_col=0
    )
    df_curated_scs = df_curated_scs
    df_curated_biowin = df_curated_biowin
    df_curated_final = df_curated_final
    return df_curated_scs, df_curated_biowin, df_curated_final


def get_dsstox(new=True) -> pd.DataFrame:
    if new:
        df_dsstox_huang = pd.read_excel("datasets/external_data/Huang_Zhang_DSStox.xlsx", index_col=0)
        df_dsstox_huang.rename(columns={"Smiles": "smiles", "CASRN": "cas"}, inplace=True)
        df_dsstox = df_dsstox_huang[["cas", "smiles"]].copy()
        df_dsstox.to_csv("datasets/external_data/DSStox.csv")
    df_dsstox = pd.read_csv("datasets/external_data/DSStox.csv", index_col=0)
    return df_dsstox


def create_fingerprint_df(df: pd.DataFrame) -> pd.DataFrame:
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    fps = [np.array(GetMACCSKeysFingerprint(mol)) for mol in mols]
    df_fp = pd.DataFrame(data=fps)
    return df_fp



def calculate_tanimoto_similarity_class(df: pd.DataFrame, model_with_best_params, nsplits=5, random_state=42):
    x = create_fingerprint_df(df=df)
    x = x.values
    y = df["y_true"]

    threshold_to_data_below: Dict[float, List[int]] = defaultdict(list)
    threshold_to_data_between: Dict[float, List[int]] = defaultdict(list)
    threshold_to_max: Dict[float, List[float]] = defaultdict(list)
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_state)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = y_train.values
        y_test = y_test.values

        model_with_best_params.fit(x_train, y_train)
        AD = ApplicabilityDomain(verbose=True)

        sims = AD.analyze_similarity(base_test=x_test, base_train=x_train,
                             similarity_metric='tanimoto')
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for index, threshold in enumerate(thresholds):
            threshold_to_data_below[threshold].append(len(sims[sims["Max"] <= threshold]))
            if index > 0:
                threshold_to_data_between[threshold].append(len(sims[(sims["Max"] <= threshold) & (sims["Max"] > thresholds[index-1])]))
            else:
                threshold_to_data_between[threshold].append(len(sims[sims["Max"] <= threshold]))

        threshold_to_value = AD.fit(
            model=model_with_best_params,
            base_test=x_test,
            base_train=x_train,
            y_true=y_test,
            threshold_reference="max",
            threshold_step=(0, 1.1, 0.1),
            similarity_metric="tanimoto",
            metric_evaliation="acc",
        )
        for threshold, value in zip(thresholds, threshold_to_value.values()):
            threshold_to_max[threshold].append(value[0][0])

    for threshold in thresholds:
        max_threshold = threshold_to_max[threshold]
        data_below = threshold_to_data_below[threshold]
        data_between = threshold_to_data_between[threshold]
        log.info(
            f"Data points below {threshold}",
            max_accuracy=f"{'%.1f' % (statistics.mean(max_threshold)*100)}"
            + " ± "
            + f"{'%.1f' % (statistics.stdev(max_threshold)*100)} %",
            datapoints_below_t=f"{'%.1f' % sum(data_below)}",
            perc_below_t=f"{'%.1f' % ((sum(data_below) / len(df)) * 100)} %",
            perc_between_t=f"{'%.1f' % ((sum(data_between) / len(df)) * 100)} %",

        )
    return


def check_external_test_in_ad(df_train: pd.DataFrame, df_test: pd.DataFrame):
    x_train = create_fingerprint_df(df=df_train)
    x_train = x_train.values
    x_test = create_fingerprint_df(df=df_test)
    x_test = x_test.values

    AD = ApplicabilityDomain(verbose=True)
    df_similarities = AD.analyze_similarity(base_test=x_test, base_train=x_train, similarity_metric="tanimoto")
    print(df_similarities.head(20))
    threshold_below_05 = len(df_similarities[df_similarities["Max"] < 0.5])
    threshold_below06 = len(df_similarities[(df_similarities["Max"] >= 0.5) & (df_similarities["Max"] < 0.6)])
    threshold_below07 = len(df_similarities[(df_similarities["Max"] >= 0.6) & (df_similarities["Max"] < 0.7)])
    threshold_below08 = len(df_similarities[(df_similarities["Max"] >= 0.7) & (df_similarities["Max"] < 0.8)])
    threshold_below09 = len(df_similarities[(df_similarities["Max"] >= 0.8) & (df_similarities["Max"] < 0.9)])
    threshold_below1 = len(df_similarities[(df_similarities["Max"] >= 0.9) & (df_similarities["Max"] < 1.0)])
    threshold_equal1 = len(df_similarities[(df_similarities["Max"] == 1.0)])
    assert ( 
        len(df_test)
        == threshold_below_05
        + threshold_below06
        + threshold_below07
        + threshold_below08
        + threshold_below09
        + threshold_below1
        + threshold_equal1
    )
    log.info(
        "Datapoints in each threshold",
        threshold_below_05=threshold_below_05,
        threshold_below06=threshold_below06,
        threshold_below07=threshold_below07,
        threshold_below08=threshold_below08,
        threshold_below09=threshold_below09,
        threshold_below1=threshold_below1,
        threshold_equal1=threshold_equal1,
    )
    log.info(
        "Percentage in each threshold",
        threshold_below_05=f"{'%.1f' % ((threshold_below_05 / len(df_test))*100)}",
        threshold_below06=f"{'%.1f' % ((threshold_below06 / len(df_test))*100)}%",
        threshold_below07=f"{'%.1f' % ((threshold_below07 / len(df_test))*100)}%",
        threshold_below08=f"{'%.1f' % ((threshold_below08 / len(df_test))*100)}%",
        threshold_below09=f"{'%.1f' % ((threshold_below09 / len(df_test))*100)}%",
        threshold_below1=f"{'%.1f' % ((threshold_below1 / len(df_test))*100)}%",
        threshold_equal1=f"{'%.1f' % ((threshold_equal1 / len(df_test))*100)}%",
    )


def calculate_tanimoto_similarity_class_huang():
    log.info("\n Define AD of classification data Huang and Zhang")
    _, _, df_class_huang = load_class_data_paper()
    df_class_huang = df_class_huang
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_class_huang, model_with_best_params=model)


def calculate_tanimoto_similarity_curated_scs():
    log.info("\n Define AD of df_curated_scs")
    df_curated_scs, _, _ = get_datasets()
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_curated_scs, model_with_best_params=model)

def calculate_tanimoto_similarity_curated_biowin():
    log.info("\n Define AD of df_curated_biowin")
    _, df_curated_biowin, _ = get_datasets()
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_curated_biowin, model_with_best_params=model)

def calculate_tanimoto_similarity_curated_final():
    log.info("\n Define AD of df_curated_final")
    _, _, df_curated_final = get_datasets()
    model = XGBClassifier()
    calculate_tanimoto_similarity_class(df=df_curated_final, model_with_best_params=model)


def check_how_much_of_dsstox_in_ad_class():
    df_dsstox = get_dsstox()
    log.info("\n Check if DSStox sets in AD of Readded classification")
    df_curated_scs, df_curated_biowin, df_curated_final = get_datasets()
    df_dsstox1 = df_dsstox[:400000]
    df_dsstox2 = df_dsstox[400000:]
    
    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_scs")
    log.info(f"\n Part 1")
    check_external_test_in_ad(df_train=df_curated_scs, df_test=df_dsstox1)
    log.info(f"\n Part 2")
    check_external_test_in_ad(df_train=df_curated_scs, df_test=df_dsstox2)

    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_biowin")
    log.info(f"\n Part 1")
    check_external_test_in_ad(df_train=df_curated_biowin, df_test=df_dsstox1)
    log.info(f"\n Part 2")
    check_external_test_in_ad(df_train=df_curated_biowin, df_test=df_dsstox2)

    log.info(f"\n                 Checking if entries of DSStox in AD of df_curated_final")
    log.info(f"\n Part 1")
    check_external_test_in_ad(df_train=df_curated_final, df_test=df_dsstox1)
    log.info(f"\n Part 2")
    check_external_test_in_ad(df_train=df_curated_final, df_test=df_dsstox2)


if __name__ == "__main__":
    calculate_tanimoto_similarity_class_huang()
    calculate_tanimoto_similarity_curated_scs()
    calculate_tanimoto_similarity_curated_biowin()
    calculate_tanimoto_similarity_curated_final()
    check_how_much_of_dsstox_in_ad_class()
