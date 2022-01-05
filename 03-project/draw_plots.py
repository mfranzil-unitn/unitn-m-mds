import json
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from pkg.logger import *


def get_outcamera_batches():
    return sorted([
        'diff_rc',
        'photoshop',
        'gimp',
        'ptlens',
        'large_scale_test_H1'
    ])


def get_model_names(with_outcamera=False):
    retval = [
        'Agfa_DC-504', 'Agfa_DC-733s', 'Agfa_DC-830i', 'Agfa_Sensor505-x', 'Agfa_Sensor530s',
        'Canon_Ixus55', 'Canon_Ixus70', 'Canon_PowerShotA640', 'Casio_EX-Z150',
        'FujiFilm_FinePixJ50', 'Kodak_M1063', 'Nikon_CoolPixS710', 'Nikon_D200', 'Nikon_D70', 'Nikon_D70s',
        'Olympus_mju_1050SW', 'Panasonic_DMC-FZ50', 'Pentax_OptioA40', 'Pentax_OptioW60', 'Praktica_DCZ5.9',
        'Ricoh_GX100', 'Rollei_RCP-7325XS', 'Samsung_L74wide', 'Samsung_NV15', 'Sony_DSC-H50',
        'Sony_DSC-T77', 'Sony_DSC-W170'
    ]

    if with_outcamera:
        retval.append('Canon-EOS-1200D')

    return sorted(retval)


def list_subfolders(folder_with_subfolders: str):
    subfolders = []

    for camera in os.listdir(folder_with_subfolders):
        if os.path.isdir(f'{folder_with_subfolders}/{camera}'):
            subfolders.append(camera)
        else:
            log.warning(f'{folder_with_subfolders}/{camera} is not a directory, skipping')

    log.info(f'Found {len(subfolders)} models: {subfolders}')

    return subfolders


def list_log_files(folder_with_files: str):
    log_files = []

    for file in os.listdir(folder_with_files):
        if file.endswith('.log'):
            log_files.append(file)

    return [i.replace('.log', '').replace('.mat', '') for i in log_files]


def generate_permutations_of_batches(batches: list):
    retval = set()

    for i in range(len(batches)):
        comb_list = combinations(batches, i + 1)
        for comb in comb_list:
            retval.add(comb)

    return list(retval)


def create_confusion_matrix(
        models: list,
        folder_with_files: str,
        batch_subfolders=None,
        outcamera=False) -> pd.DataFrame:
    """
    Creates a confusion matrix for each model in the folder_with_files.
    Accepts a list of batch subfolders to filter the data (i.e. if I don't want to
    analyze some of the data).
    :param models: list of models to add in the matrix
    :param folder_with_files: The folder with the files to analyze
    :param batch_subfolders: A list of subfolders to filter the data
    :param outcamera: If True, then all photos are taken by the Canon EOS 1200D (outcamera dataset)
    :return: a pandas dataframe with the confusion matrix
    """
    if outcamera:
        raise NotImplementedError('Outcamera dataset not implemented yet')

    if batch_subfolders is None:
        batch_subfolders = list_log_files(folder_with_files)

    targets = []
    predictions = []

    for __class in list_log_files(folder_with_files):
        if __class not in batch_subfolders:
            # If the class is not in the list, then we don't want to analyze it
            continue

        try:
            with open(f'{folder_with_files}/{__class}.json', 'r') as f:
                classification = json.load(f)
                # "h0": {
                #     "Kodak_M1063_4_12092.mat": {
                #         "template": {
                #             "name": "/Users/matte/Downloads/mds/fingerprints/Agfa_DC-504.mat",
                #             "width": 4032,
                #             "height": 3024
                #         },
                #         "image": {
                #             "name": "/Volumes/Extreme SSD/_move/dataset-dresden-validation-noise/Kodak_M1063/Kodak_M1063_4_12092.mat",
                #             "width": 2748,
                #             "height": 3664
                #         },
                #         "pce": 0.45516441817269054,
                #         "ncc": 0.08586090803146362,
                #         "distance": 665.472900390625
                #     },
                h1 = classification['h1']
                h0 = classification['h0']

                for item in classification:
                    targets.append(__class)
                    predictions.append(item['template'])
        except FileNotFoundError:
            log.warning(f'{folder_with_files}/{__class}.json not found, skipping')

    conf_mat = confusion_matrix(targets, predictions, labels=models)
    pd.DataFrame(conf_mat, index=models, columns=models)

    # Plot a heatmap of the confusion matrix, with labels and centering it
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=models, yticklabels=models)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # with open(f'{folder_with_files}/confusion_matrix.json', 'w') as f:
    #    json.dump(confusion_matrix, f)

    return conf_mat


def create_roc_curve(
        folder_with_files: str,
        batch_subfolders=None,
        outcamera=False) -> {}:
    if outcamera:
        no_subtitle = False
    else:
        if batch_subfolders is None:
            # If no list is provided, we list all the available log files
            no_subtitle = True
            batch_subfolders = list_log_files(folder_with_files)
        else:
            no_subtitle = False

    scores = []
    labels = []

    # Structure of the JSON files
    # "h0": {
    #         "image_name.mat": {
    #             "template": { "name": "template_name", "width": ..., "height": ... },
    #             "image": { "name": "image_name", "width": ..., "height": ... },
    #             "pce": pce, "ncc": ncc, "distance": distance
    #       },
    # },
    # "h1": { ... }

    if outcamera:  # == just Canon-EOS-1200D log
        # First assert the log is just one
        if len(list_log_files(folder_with_files)) != 1:
            raise ValueError("More than a JSON file present in the folder.")

        logname = list_log_files(folder_with_files)[0]
        filename = f'{folder_with_files}/{logname}.json'

        with open(filename, 'r') as f:
            classification = json.load(f)

            # H1's a little different this time
            # We need to check if we can actually count in each file
            # Then we balance the datasets
            h1 = classification['h1']
            for image_name in h1:
                ori_image_filename = h1[image_name]["image"]["name"]
                # Check if the result is processable
                processable = False
                for batch in batch_subfolders:
                    if batch in ori_image_filename.split("/"):
                        processable = True
                        break

                if processable:
                    labels.append(1)
                    scores.append(h1[image_name]['ncc'])

            h0 = classification['h0']
            for image_name in random.sample(h0.keys(), len(labels)):
                labels.append(0)
                scores.append(h0[image_name]['ncc'])

    else:
        for __class in list_log_files(folder_with_files):
            if __class not in batch_subfolders:
                # If the class is not in the list, then we don't want to analyze it
                continue

            filename = f'{folder_with_files}/{__class}.json'
            try:
                with open(filename, 'r') as f:
                    classification = json.load(f)
                    h0 = classification['h0']
                    for image_name in h0:
                        labels.append(0)
                        scores.append(h0[image_name]['ncc'])

                    h1 = classification['h1']
                    for image_name in h1:
                        labels.append(1)
                        scores.append(h1[image_name]['ncc'])

            except FileNotFoundError:
                log.warning(f'{filename} not found, skipping')

    # h1__ = sorted([scores[i] for i in range(len(scores)) if labels[i] == 1])
    # h0__ = sorted([scores[i] for i in range(len(scores)) if labels[i] == 0])
    # print(len(h1__), len(h0__))
    #
    # plt.plot(range(len(h1__)+len(h0__)), (h1__+h0__), label='h1+h0',
    #           color='red')
    # plt.title(f'H1+H0 for {" ".join(batch_subfolders)}')
    # plt.show()

    # Plot the ROC curve with roc_curve and auc (code from lab)
    fpr, tpr, tau = roc_curve(labels, scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 12))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    if roc_auc < 0 or roc_auc > 1:
        raise ValueError("AUC out of bounds")
    elif roc_auc == 0 or roc_auc < 0.001:
        output_roc = "__0000"
    elif roc_auc == 1 or roc_auc > 0.999:
        output_roc = "__perf"
    else:
        output_roc = f'{roc_auc:.4f}'.replace('0.', '__')

    filename = f'{folder_with_files}/{output_roc}'
    if no_subtitle:
        plt.title('ROC (all batches)')
        filename += f'-all.png'
    else:
        plt.title('ROC of {}'.format(", ".join(batch_subfolders)))
        filename += f'-{"-".join(batch_subfolders)}.png'

    plt.savefig(filename)

    # return pd.DataFrame({"fpr": fpr, "tpr": tpr, "tau": tau})
    return {
        "experiment": folder_with_files + "-" + "+".join(batch_subfolders),
        "outcamera": outcamera,
        "scores": scores,
        "labels": labels,
    }


def get_optimal_threshold(scores: [], labels: []) -> (float, int):
    fpr, tpr, tau = roc_curve(labels, scores, drop_intermediate=False)
    # Get the optimal threshold
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(tau, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])[0], int(roc_t.axes[0][0]), fpr[int(roc_t.axes[0][0])]


def create_confusion_table(exps: list) -> pd.DataFrame:
    """
    Create a TPR/FPR table from a list of dictionaries.
    Expected form of the dictionary:
    [{
        "experiment": "",
        "outcamera": "",
        "scores": [],
        "labels": []
    }, ...]
    """

    all_scores = []
    all_labels = []
    for item in exps:
        all_scores.extend(item['scores'])
        all_labels.extend(item['labels'])

    optimal_threshold, optimal_index, optimal_fpr = get_optimal_threshold(all_scores, all_labels)

    print(f"Alternate thresholds for {len(exps)} experiments:")
    print(f"{optimal_threshold:.4f} (optimal, {optimal_fpr:.4f} FPR)")
    fpr, tpr, tau = roc_curve(np.asarray(all_labels), np.asarray(all_scores), drop_intermediate=False)
    idx_tpr20 = np.where((fpr - 0.2) == min(i for i in (fpr - 0.2) if i > 0))
    print(f"{tau[idx_tpr20[0][0]]:.4f} (0.2 FPR)")
    idx_tpr10 = np.where((fpr - 0.1) == min(i for i in (fpr - 0.1) if i > 0))
    print(f"{tau[idx_tpr10[0][0]]:.4f} (0.1 FPR)")
    idx_tpr5 = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))
    print(f"{tau[idx_tpr5[0][0]]:.4f} (0.05 FPR)")

    conf_arr = []
    experiment_labels = []
    for experiment in exps:
        experiment_labels.append(experiment["experiment"]
                                 .split("/")[-1]
                                 .replace('dataset-dresden-results-', '')
                                 .replace('dataset-outcamera-results-', ' '))
        scores = experiment['scores']

        tpr_count = len([i for i in scores if i >= optimal_threshold]) / len(scores)
        fpr_count = len([i for i in scores if i < optimal_threshold]) / len(scores)
        #print(f"tpr: {tpr_count:.4f}, ratio of {len([i for i in scores if i >= optimal_threshold])}/{len(scores)}")
        #print(f"fpr: {fpr_count:.4f}, ratio of {len([i for i in scores if i < optimal_threshold])}/{len(scores)}")
        conf_arr.append({"tpr": tpr_count, "fpr": fpr_count})

    # Create the table
    conf_table = pd.DataFrame(conf_arr)

    # Plot a heatmap of the confusion matrix, with labels and centering it
    plt.figure(figsize=(5, 7.5))
    sns.heatmap(conf_table, annot=True, cmap='Reds',
                xticklabels=['TPR', 'FPR'], yticklabels=experiment_labels, cbar=False)
    plt.ylabel('Class')
    plt.xlabel(f'(threshold: {optimal_threshold:.2f})')
    plt.tight_layout()
    #plt.savefig(f'{"-".join(i[:4] for i in experiment_labels)}.png')
    plt.show()

    return conf_table


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    DRESDEN_FODLER = '/Users/matte/Downloads/mds/dataset-dresden-results'
    OUTCAMERA_FOLDER = '/Users/matte/Downloads/mds/dataset-outcamera-results'

    create_roc = False
    create_ctb = True

    if create_ctb:
        # Create a "TPR/FPR" table for the outcamera results
        json = json.loads(open("/Users/matte/OneDrive/Codice/github/unitn-m-mds/project/results/logs/results.json", "r").read())
        outcamera = [
            x
            for x in json
            if x["outcamera"] == True
               # and 'photoshop' not in x['experiment']
        ]
        dresdy = [
            x
            for x in json
            if x["outcamera"] == False
               # and 'D200' not in x['experiment']
               # and '530s' not in x['experiment']
               # and '504' not in x['experiment']
               # and 'D70' not in x['experiment']
        ]
        create_confusion_table(outcamera)
        create_confusion_table(dresdy)
        create_confusion_table(outcamera + dresdy)

    if create_roc:
        results = []
        outcamera_batches = [
            (i,)
            for i in get_outcamera_batches()
        ]
        dresden_batches = [
            (i,)
            for i in get_model_names()
        ]
        # batches  = generate_permutations_of_batches(get_outcamera_batches())
        for __list in outcamera_batches:
            result = create_roc_curve(
                batch_subfolders=__list,
                folder_with_files=OUTCAMERA_FOLDER,
                outcamera=True
            )
            results.append(result)
        for __list in dresden_batches:
            result = create_roc_curve(
                batch_subfolders=__list,
                folder_with_files=DRESDEN_FODLER,
                outcamera=False
            )
            results.append(result)

        json.dump(results, open("results-total.json", "w"))
