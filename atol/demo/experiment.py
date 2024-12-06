import os
import numpy as np
from itertools import product
import sys
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path
parent_dir = os.path.join(script_dir, '..')
sys.path.append(parent_dir)
from atol.utils import compute_tda_for_graphs, graph_dtypes, csv_toarray, atol_feats_graphs, graph_tenfold

from atol.atol import Atol

# Parameters
graph_problems = ["BZR", "COX2", "DHFR", "FRANKENSTEIN", "IMDB-BINARY", "IMDB-MULTI", "IMDB-NCI1", "MUTAG", "NCI1", "NCI109", "PROTEINS"]
graph_folder_base = "../../perslay/tutorial/data/"
filtrations = ['0.1-hks', '10.0-hks']
num_repeats = 10

# DataFrame to store results
results_df = pd.DataFrame(columns=["Graph Problem", "Avg Score", "Std Dev Score", "Avg Feature Time (ms)", "Std Dev Feature Time (ms)"])

for graph_problem in graph_problems:
    graph_folder = graph_folder_base + graph_problem + "/"

    # TDA Computation
    print(f"- [{graph_problem}] TDA computation")
    compute_tda_for_graphs(graph_folder=graph_folder, filtrations=filtrations)

    # Load Graph Data
    num_elements = len(os.listdir(graph_folder + "mat/"))
    all_diags = {(dtype, filt, gid): csv_toarray(graph_folder + f"diagrams/{dtype}/graph_{gid:06d}_filt_{filt}.csv")
                 for dtype, gid, filt in product(graph_dtypes, np.arange(num_elements), filtrations)}

    # Run Experiment Multiple Times to Calculate Standard Deviation
    vfold_scores_all = []
    feature_times_all = []

    for repeat in range(num_repeats):
        print(f"- [{graph_problem}] Run {repeat + 1}/{num_repeats}")

        # ATOL Objects and Fitting
        atol_objs = {(dtype, filt): Atol(quantiser=MiniBatchKMeans(n_clusters=10)) for dtype, filt in product(graph_dtypes, filtrations)}
        for dtype, filt in product(graph_dtypes, filtrations):
            atol_objs[(dtype, filt)].fit([all_diags[(dtype, filt, gid)] for gid in np.arange(num_elements)])

        # Feature Extraction and Cross-validation
        vfold_scores, feature_times = graph_tenfold(graph_folder, filtrations)
        vfold_scores_all.append(np.mean(vfold_scores))
        feature_times_all.append(np.mean(feature_times))

        # Store Results in DataFrame
        new_row = pd.DataFrame([{
        "Graph Problem": graph_problem,
        "Avg Score": np.mean(vfold_scores_all),
        "Std Dev Score": np.std(vfold_scores_all),
        "Avg Feature Time (ms)": np.mean(feature_times_all) * 1000.0,
        "Std Dev Feature Time (ms)": np.std(feature_times_all) * 1000.0
    }])

        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Print and Save Results
    print(results_df)
    results_df.to_csv("tda_experiment_results.csv", index=False)
    print(f"- All experiments completed and results saved to 'tda_experiment_results.csv'\n")
