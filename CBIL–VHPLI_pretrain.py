#!/usr/bin/env python
# coding: utf-8
import math
import os
import gc
import random
from itertools import chain, product

import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# os.chdir('/pycharm_project/lncRNA相互作用')
# print(os.getcwd())


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def rna_one_hot(seqs, rna_vocab_to_int, rna_vocab_size):
    k = 4
    max_len = max_length_rna
    encodings = []
    for rna_seq in seqs:
        one_hot = np.zeros((len(rna_seq), rna_vocab_size), dtype=np.int8)
        for i in range(len(rna_seq) - k + 1):
            kmer = rna_seq[i: i + k]
            if kmer in rna_vocab_to_int:
                one_hot[i: i + k, rna_vocab_to_int[kmer]] = 1
        one_hot = pad_sequences(np.array(one_hot).T, maxlen=max_len, dtype="int8")
        encodings.append(one_hot)
    #         print(one_hot.shape)
    return np.array(encodings)


def protein_one_hot(seqs, protein_vocab_to_int, protein_vocab_size):
    k = 3
    max_len = max_length_pro
    encodings = []
    for pr_seq in seqs:
        one_hot = np.zeros((len(pr_seq), protein_vocab_size), dtype=np.int8)
        for i in range(len(pr_seq) - k + 1):
            kmer = pr_seq[i: i + k]
            if kmer in protein_vocab_to_int:
                one_hot[i: i + k, protein_vocab_to_int[kmer]] = 1
        one_hot = pad_sequences(np.array(one_hot).T, maxlen=max_len, dtype="int8")
        encodings.append(one_hot)
    #         print(one_hot.shape)
    return np.array(encodings)


# def encode_data(data):
#     rna_seq = data["RNA_sequence"]
#     bases = ["A", "C", "G", "U"]
#     rna_k = 4
#     rna_vocab = ["".join(p) for p in product(bases, repeat=rna_k)]
#     rna_vocab_size = len(rna_vocab)
#     rna_vocab_to_int = {kmer: i for i, kmer in enumerate(rna_vocab)}
#     rna_one_hot_X = np.array([rna_one_hot(rna_seq, rna_vocab_to_int, rna_vocab_size)])
#     del rna_seq
#     gc.collect()
#
#     protein_seq = data["protein_sequence"]
#     amino_acids = [
#         "A",
#         "C",
#         "D",
#         "E",
#         "F",
#         "G",
#         "H",
#         "I",
#         "K",
#         "L",
#         "M",
#         "N",
#         "P",
#         "Q",
#         "R",
#         "S",
#         "T",
#         "V",
#         "W",
#         "Y",
#     ]
#     pro_k = 3
#     protein_vocab = ["".join(p) for p in product(amino_acids, repeat=pro_k)]
#     protein_vocab_size = len(protein_vocab)
#     protein_vocab_to_int = {kmer: i for i, kmer in enumerate(protein_vocab)}
#     protein_one_hot_X = np.array(
#         [protein_one_hot(protein_seq, protein_vocab_to_int, protein_vocab_size)]
#     )
#     del protein_seq
#     gc.collect()
#     print(rna_one_hot_X.shape, protein_one_hot_X.shape)
#
#     return (
#         rna_one_hot_X,
#         protein_one_hot_X,
#         np.array(data["label"]),
#         rna_vocab_size,
#         protein_vocab_size,
#         rna_vocab_to_int,
#         protein_vocab_to_int,
#     )

# 7分类
def encode_data(data):
    rna_seq = data["RNA_sequence"]
    bases = ["A", "C", "G", "U"]
    rna_k = 4
    rna_vocab = ["".join(p) for p in product(bases, repeat=rna_k)]
    rna_vocab_size = len(rna_vocab)
    rna_vocab_to_int = {kmer: i for i, kmer in enumerate(rna_vocab)}
    rna_one_hot_X = np.array([rna_one_hot(rna_seq, rna_vocab_to_int, rna_vocab_size)])
    del rna_seq
    gc.collect()

    protein_seq = data["protein_sequence"]
    amino_acid_classes = {
        'A': '1', 'G': '1', 'V': '1',
        'I': '2', 'L': '2', 'F': '2', 'P': '2',
        'Y': '3', 'M': '3', 'T': '3', 'S': '3',
        'H': '4', 'N': '4', 'Q': '4', 'W': '4',
        'R': '5', 'K': '5',
        'D': '6', 'E': '6',
        'C': '7'
    }
    # amino_acid_seq = "".join([amino_acid_classes[aa] for aa in protein_seq])
    amino_acid_classes_unique = list(set(amino_acid_classes.values()))
    pro_k = 3
    protein_vocab = ["".join(p) for p in product(amino_acid_classes_unique, repeat=pro_k)]
    protein_vocab_size = len(protein_vocab)
    protein_vocab_to_int = {kmer: i for i, kmer in enumerate(protein_vocab)}
    protein_one_hot_X = np.array(
        [protein_one_hot(protein_seq, protein_vocab_to_int, protein_vocab_size)]
    )
    del protein_seq
    gc.collect()
    print(rna_one_hot_X.shape, protein_one_hot_X.shape)

    return (
        rna_one_hot_X,
        protein_one_hot_X,
        np.array(data["label"]),
        rna_vocab_size,
        protein_vocab_size,
        rna_vocab_to_int,
        protein_vocab_to_int,
    )

def cal_scores(
        model_name, y_test, y_pred, csv_path="./results/before_fintue_results.csv"
):
    results = {}
    pred_test = [1 if i > 0.5 else 0 for i in y_pred]
    results["accuracy"] = accuracy_score(y_test, pred_test)
    results["roc_score"] = roc_auc_score(y_test, y_pred)
    results["recall"] = recall_score(y_test, pred_test)
    results["precision"] = precision_score(y_test, pred_test)
    results["f1"] = f1_score(y_test, pred_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    results["specificity"] = tn / (tn + fp)
    results["ppv"] = tp / (tp + fp)
    results["prc_score"] = average_precision_score(y_test, y_pred)
    results["mcc"] = matthews_corrcoef(y_test, y_pred)
    print(results)
    with open(csv_path, "a") as f:
        f.write(
            f"{model_name},{results['accuracy']},{results['roc_score']},{results['recall']},{results['precision']},{results['f1']},{results['specificity']},{results['ppv']},{results['prc_score']}\n"
        )


def plot_acc_loss(history, model_name):
    # Plot the training and validation loss
    plt.plot(history.history["loss"])
    if not model_name.endswith("_finetune"):
        plt.plot(history.history["val_loss"])
        plt.legend(["Train", "Validation"])
    else:
        plt.legend(["Train"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(fname=f"./figs/{model_name}_loss.png")
    plt.show()
    plt.close()

    # Plot the training and validation accuracy
    plt.plot(history.history["accuracy"])
    if not model_name.endswith("_finetune"):
        plt.plot(history.history["val_accuracy"])
        plt.legend(["Train", "Validation"])
    else:
        plt.legend(["Train"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(fname=f"./figs/{model_name}_accuracy.png")
    plt.show()
    plt.close()

    # Plot the training and validation accuracy
    plt.plot(history.history["auc"])
    if not model_name.endswith("_finetune"):
        plt.plot(history.history["val_auc"])
        plt.legend(["Train", "Validation"])
    else:
        plt.legend(["Train"])
    plt.title("Model AUC")
    plt.ylabel("AUC")
    plt.xlabel("Epoch")
    plt.savefig(fname=f"./figs/{model_name}_auc.png")
    plt.show()
    plt.close()


def define_model(model_name, max_length_rna, max_length_pro, rna_vocab_size, protein_vocab_size):
    # Build the model
    if model_name == "CNN":
        model1 = keras.Sequential(
            [
                keras.layers.Conv1D(
                    64,
                    3,
                    padding="same",
                    activation="relu",
                    input_shape=(max_length_rna, rna_vocab_size),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        model2 = keras.Sequential(
            [
                keras.layers.Conv1D(
                    64,
                    5,
                    padding="same",
                    activation="relu",
                    input_shape=(max_length_pro, protein_vocab_size),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )

        model3 = keras.Sequential(
            [
                keras.layers.Reshape((144, 1), input_shape=(144,)),
                keras.layers.Conv1D(
                    64,
                    3,
                    padding="same",
                    activation="relu",
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        model4 = keras.Sequential(
            [
                keras.layers.Reshape((195, 1), input_shape=(195,)),
                keras.layers.Conv1D(
                    64,
                    5,
                    padding="same",
                    activation="relu",
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )

        merged = keras.layers.concatenate(
            [model1.output, model2.output, model3.output, model4.output]
        )
        mlp = keras.layers.Dense(64, activation="relu")(merged)
        output = keras.layers.Dense(1, activation="sigmoid")(mlp)
        model = keras.Model(
            inputs=[model1.input, model2.input, model3.input, model4.input],
            outputs=output,
        )

    elif model_name == "lstm":
        model1 = keras.Sequential(
            [
                keras.layers.LSTM(
                    64,
                    return_sequences=True,
                    input_shape=(max_length_rna, rna_vocab_size),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        model2 = keras.Sequential(
            [
                keras.layers.LSTM(
                    64,
                    return_sequences=True,
                    input_shape=(max_length_pro, protein_vocab_size),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        model3 = keras.Sequential(
            [
                keras.layers.Reshape((144, 1), input_shape=(144,)),
                keras.layers.LSTM(
                    64,
                    return_sequences=True,
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        model4 = keras.Sequential(
            [
                keras.layers.Reshape((195, 1), input_shape=(195,)),
                keras.layers.LSTM(
                    64,
                    return_sequences=True,
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        merged = keras.layers.concatenate(
            [model1.output, model2.output, model3.output, model4.output]
        )
        mlp = keras.layers.Dense(64, activation="relu")(merged)
        output = keras.layers.Dense(1, activation="sigmoid")(mlp)
        model = keras.Model(
            inputs=[model1.input, model2.input, model3.input, model4.input],
            outputs=output,
        )

    elif model_name == "rnn":
        model1 = keras.Sequential(
            [
                keras.layers.SimpleRNN(  # changed LSTM to SimpleRNN
                    64,
                    return_sequences=True,
                    input_shape=(max_length_rna, rna_vocab_size),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        model2 = keras.Sequential(
            [
                keras.layers.SimpleRNN(  # changed LSTM to SimpleRNN
                    64,
                    return_sequences=True,
                    input_shape=(max_length_pro, protein_vocab_size),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        model3 = keras.Sequential(
            [
                keras.layers.Reshape((144, 1), input_shape=(144,)),
                keras.layers.SimpleRNN(  # changed LSTM to SimpleRNN
                    64,
                    return_sequences=True,
                    input_shape=(144,),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        model4 = keras.Sequential(
            [
                keras.layers.Reshape((195, 1), input_shape=(195,)),
                keras.layers.SimpleRNN(  # changed LSTM to SimpleRNN
                    64,
                    return_sequences=True,
                    input_shape=(195,),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
            ]
        )
        merged = keras.layers.concatenate(
            [model1.output, model2.output, model3.output, model4.output]
        )
        mlp = keras.layers.Dense(64, activation="relu")(merged)
        output = keras.layers.Dense(1, activation="sigmoid")(mlp)
        model = keras.Model(
            inputs=[model1.input, model2.input, model3.input, model4.input],
            outputs=output,
        )

    elif model_name == "bilstm":
        model1 = keras.Sequential(
            [
                keras.layers.Bidirectional(
                    keras.layers.LSTM(64, return_sequences=True),
                    input_shape=(max_length_rna, rna_vocab_size),
                ),
                keras.layers.Flatten(),
            ]
        )
        model2 = keras.Sequential(
            [
                keras.layers.Bidirectional(
                    keras.layers.LSTM(64, return_sequences=True),
                    input_shape=(max_length_pro, protein_vocab_size),
                ),
                keras.layers.Flatten(),
            ]
        )
        model3 = keras.Sequential(
            [
                keras.layers.Bidirectional(
                    keras.layers.LSTM(64, return_sequences=True),
                    input_shape=(144, 1),
                ),
                keras.layers.Flatten(),
            ]
        )
        model4 = keras.Sequential(
            [
                keras.layers.Bidirectional(
                    keras.layers.LSTM(64, return_sequences=True),
                    input_shape=(195, 1),
                ),
                keras.layers.Flatten(),
            ]
        )
        merged = keras.layers.concatenate(
            [model1.output, model2.output, model3.output, model4.output]
        )
        mlp = keras.layers.Dense(64, activation="relu")(merged)
        output = keras.layers.Dense(1, activation="sigmoid")(mlp)
        model = keras.Model(
            inputs=[model1.input, model2.input, model3.input, model4.input],
            outputs=output,
        )

    elif model_name == "CBIL–VHPLI":
        model1 = keras.Sequential(
            [
                keras.layers.Conv1D(
                    64,
                    3,
                    padding="same",
                    activation="relu",
                    input_shape=(max_length_rna, rna_vocab_size),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Bidirectional(keras.layers.LSTM(32)),
            ]
        )
        model2 = keras.Sequential(
            [
                keras.layers.Conv1D(
                    64,
                    5,
                    padding="same",
                    activation="relu",
                    input_shape=(max_length_pro, protein_vocab_size),
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Bidirectional(keras.layers.LSTM(32)),
            ]
        )
        model3 = keras.Sequential(
            [
                keras.layers.Reshape((144, 1), input_shape=(144,)),
                keras.layers.Conv1D(
                    64,
                    3,
                    padding="same",
                    activation="relu",
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Bidirectional(keras.layers.LSTM(32)),
            ]
        )
        model4 = keras.Sequential(
            [
                keras.layers.Reshape((195, 1), input_shape=(195,)),
                keras.layers.Conv1D(
                    64,
                    5,
                    padding="same",
                    activation="relu",
                ),
                keras.layers.MaxPooling1D(2),
                keras.layers.Bidirectional(keras.layers.LSTM(32)),
            ]
        )
        merged = keras.layers.concatenate(
            [model1.output, model2.output, model3.output, model4.output]
        )
        mlp = keras.layers.Dense(64, activation="relu")(merged)
        output = keras.layers.Dense(1, activation="sigmoid")(mlp)
        model = keras.Model(
            inputs=[model1.input, model2.input, model3.input, model4.input],
            outputs=output,
        )

    return model


def train_all_models(
        rna_one_hot_X_train,
        protein_one_hot_X_train,
        rna_encoding_train,
        pro_encoding_train,
        rna_one_hot_X_val,
        protein_one_hot_X_val,
        rna_encoding_val,
        pro_encoding_val,
        y_train,
        y_val,
        max_length_rna,
        max_length_pro,
        rna_vocab_size,
        protein_vocab_size,
        lr,
        epochs
):
    suffix = "pretrained"
    csv_path = "./results/before_fintue_results.csv"
    for model_name in ['CNN', 'lstm', 'rnn', 'bilstm', 'CBIL–VHPLI']:
        model = define_model(model_name, max_length_rna, max_length_pro, rna_vocab_size, protein_vocab_size)
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=lr),
            metrics=["accuracy", "AUC"],
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"models/{model_name + suffix}.h5",
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
        )
        print(f"Training model {model_name}...")
        history = model.fit(
            (
                rna_one_hot_X_train,
                protein_one_hot_X_train,
                rna_encoding_train,
                pro_encoding_train,
            ),
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                (
                    rna_one_hot_X_val,
                    protein_one_hot_X_val,
                    rna_encoding_val,
                    pro_encoding_val,
                ),
                y_val,
            ),
            callbacks=[checkpoint],
        )
        model.save_weights(f'models/{model_name + suffix}.h5')
        model.load_weights(f"models/{model_name + suffix}.h5")
        if not os.path.isfile(csv_path):
            with open(csv_path, "w") as f:
                f.write(
                    "Model,accuracy,roc_score,recall,precision,f1,specificity,ppv,prc_score\n"
                )
        cal_scores(
            model_name + suffix,
            y_val,
            model.predict(
                (
                    rna_one_hot_X_val,
                    protein_one_hot_X_val,
                    rna_encoding_val,
                    pro_encoding_val,
                )
            ),
        )
        plot_acc_loss(history, model_name + suffix)


def Protein_CTDD(data):
    #     https://github.com/Superzchen/iLearnPlus/blob/main/util/FileProcessing.py
    group1 = {
        "hydrophobicity_PRAM900101": "RKEDQN",
        "hydrophobicity_ARGP820101": "QSTNGDE",
        "hydrophobicity_ZIMJ680101": "QNGSWTDERA",
        "hydrophobicity_PONP930101": "KPDESNQT",
        "hydrophobicity_CASG920101": "KDEQPSRNTG",
        "hydrophobicity_ENGD860101": "RDKENQHYP",
        "hydrophobicity_FASG890101": "KERSQD",
        "normwaalsvolume": "GASTPDC",
        "polarity": "LIFWCMVY",
        "polarizability": "GASDT",
        "charge": "KR",
        "secondarystruct": "EALMQKRH",
        "solventaccess": "ALFCGIVW",
    }
    group2 = {
        "hydrophobicity_PRAM900101": "GASTPHY",
        "hydrophobicity_ARGP820101": "RAHCKMV",
        "hydrophobicity_ZIMJ680101": "HMCKV",
        "hydrophobicity_PONP930101": "GRHA",
        "hydrophobicity_CASG920101": "AHYMLV",
        "hydrophobicity_ENGD860101": "SGTAW",
        "hydrophobicity_FASG890101": "NTPG",
        "normwaalsvolume": "NVEQIL",
        "polarity": "PATGS",
        "polarizability": "CPNVEQIL",
        "charge": "ANCQGHILMFPSTWYV",
        "secondarystruct": "VIYCWFT",
        "solventaccess": "RKQEND",
    }
    group3 = {
        "hydrophobicity_PRAM900101": "CLVIMFW",
        "hydrophobicity_ARGP820101": "LYPFIW",
        "hydrophobicity_ZIMJ680101": "LPFYI",
        "hydrophobicity_PONP930101": "YMFWLCVI",
        "hydrophobicity_CASG920101": "FIWC",
        "hydrophobicity_ENGD860101": "CVLIMF",
        "hydrophobicity_FASG890101": "AYHWVMFLIC",
        "normwaalsvolume": "MHKFRYW",
        "polarity": "HQRKNED",
        "polarizability": "KMHFRYW",
        "charge": "DE",
        "secondarystruct": "GNPSD",
        "solventaccess": "MSPTHY",
    }

    groups = [group1, group2, group3]
    property = (
        "hydrophobicity_PRAM900101",
        "hydrophobicity_ARGP820101",
        "hydrophobicity_ZIMJ680101",
        "hydrophobicity_PONP930101",
        "hydrophobicity_CASG920101",
        "hydrophobicity_ENGD860101",
        "hydrophobicity_FASG890101",
        "normwaalsvolume",
        "polarity",
        "polarizability",
        "charge",
        "secondarystruct",
        "solventaccess",
    )

    encodings = []
    header = ["protein", "label"]
    for p in property:
        for g in ("1", "2", "3"):
            for d in ["0", "25", "50", "75", "100"]:
                header.append(p + "." + g + ".residue" + d)
    encodings.append(header)
    print(data)
    for i, row in data.iterrows():
        sequence = row["protein_sequence"]
        name, label = row["protein"], row["label"]
        code = [name, label]
        for p in property:
            code = (
                    code
                    + Count1(group1[p], sequence)
                    + Count1(group2[p], sequence)
                    + Count1(group3[p], sequence)
            )
        encodings.append(code)

    encoding_array = np.array([])
    encoding_array = np.array(encodings, dtype=str)
    column = encoding_array.shape[1]
    row = encoding_array.shape[0] - 1
    del encodings
    if encoding_array.shape[0] > 1:
        return encoding_array
    else:
        return False


def Count1(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [
        1,
        math.floor(0.25 * number),
        math.floor(0.50 * number),
        math.floor(0.75 * number),
        number,
    ]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code


def Z_curve_144bit(data):
    NN = "ACGU"
    encodings = []
    header = ["SampleName", "label"]

    for base in NN:
        for base1 in NN:
            for pos in range(1, 4):
                for elem in ["x", "y", "z"]:
                    header.append("Pos_%s_%s%s.%s" % (pos, base, base1, elem))
    encodings.append(header)

    for i, row in data.iterrows():
        sequence = row["protein_sequence"]
        name, label = row["protein"], row["label"]
        code = [name, label]
        pos1_dict = {}
        pos2_dict = {}
        pos3_dict = {}
        for i in range(len(sequence) - 2):
            if (i + 1) % 3 == 1:
                if sequence[i: i + 3] in pos1_dict:
                    pos1_dict[sequence[i: i + 3]] += 1
                else:
                    pos1_dict[sequence[i: i + 3]] = 1
            elif (i + 1) % 3 == 2:
                if sequence[i: i + 3] in pos2_dict:
                    pos2_dict[sequence[i: i + 3]] += 1
                else:
                    pos2_dict[sequence[i: i + 3]] = 1
            elif (i + 1) % 3 == 0:
                if sequence[i: i + 3] in pos3_dict:
                    pos3_dict[sequence[i: i + 3]] += 1
                else:
                    pos3_dict[sequence[i: i + 3]] = 1

        for base in NN:
            for base1 in NN:
                code += [
                    (
                            pos1_dict.get("%s%sA" % (base, base1), 0)
                            + pos1_dict.get("%s%sG" % (base, base1), 0)
                            - pos1_dict.get("%s%sC" % (base, base1), 0)
                            - pos1_dict.get("%s%sU" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # x
                    (
                            pos1_dict.get("%s%sA" % (base, base1), 0)
                            + pos1_dict.get("%s%sC" % (base, base1), 0)
                            - pos1_dict.get("%s%sG" % (base, base1), 0)
                            - pos1_dict.get("%s%sU" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # y
                    (
                            pos1_dict.get("%s%sA" % (base, base1), 0)
                            + pos1_dict.get("%s%sU" % (base, base1), 0)
                            - pos1_dict.get("%s%sG" % (base, base1), 0)
                            - pos1_dict.get("%s%sC" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # z
                ]
                code += [
                    (
                            pos2_dict.get("%s%sA" % (base, base1), 0)
                            + pos2_dict.get("%s%sG" % (base, base1), 0)
                            - pos2_dict.get("%s%sC" % (base, base1), 0)
                            - pos2_dict.get("%s%sU" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # x
                    (
                            pos2_dict.get("%s%sA" % (base, base1), 0)
                            + pos2_dict.get("%s%sC" % (base, base1), 0)
                            - pos2_dict.get("%s%sG" % (base, base1), 0)
                            - pos2_dict.get("%s%sU" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # y
                    (
                            pos2_dict.get("%s%sA" % (base, base1), 0)
                            + pos2_dict.get("%s%sU" % (base, base1), 0)
                            - pos2_dict.get("%s%sG" % (base, base1), 0)
                            - pos2_dict.get("%s%sC" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # z
                ]
                code += [
                    (
                            pos3_dict.get("%s%sA" % (base, base1), 0)
                            + pos3_dict.get("%s%sG" % (base, base1), 0)
                            - pos3_dict.get("%s%sC" % (base, base1), 0)
                            - pos3_dict.get("%s%sU" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # x
                    (
                            pos3_dict.get("%s%sA" % (base, base1), 0)
                            + pos3_dict.get("%s%sC" % (base, base1), 0)
                            - pos3_dict.get("%s%sG" % (base, base1), 0)
                            - pos3_dict.get("%s%sU" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # y
                    (
                            pos3_dict.get("%s%sA" % (base, base1), 0)
                            + pos3_dict.get("%s%sU" % (base, base1), 0)
                            - pos3_dict.get("%s%sG" % (base, base1), 0)
                            - pos3_dict.get("%s%sC" % (base, base1), 0)
                    )
                    / (len(sequence) - 2),  # z
                ]
        encodings.append(code)

    encoding_array = np.array([])
    encoding_array = np.array(encodings, dtype=str)
    column = encoding_array.shape[1]
    row = encoding_array.shape[0] - 1
    del encodings
    if encoding_array.shape[0] > 1:
        return encoding_array
    else:
        return False


def cross_validation(rna_one_hot_X_train, protein_one_hot_X_train, rna_encoding_train, pro_encoding_train,
                     rna_one_hot_X_val, protein_one_hot_X_val, rna_encoding_val, pro_encoding_val, y_train, y_val,
                     max_length, rna_vocab_size, protein_vocab_size, lr, dataset, epochs):
    suffix = 'cv'
    csv_path = './results/cross_validation_results.csv'
    for model_name in ['CNN', 'lstm', 'rnn', 'bilstm', 'CBIL–VHPLI']:
        model = define_model(model_name, max_length, rna_vocab_size, protein_vocab_size)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy', 'AUC'])
        model.load_weights(f'models/{model_name}pretrained.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'models/{model_name + suffix}.h5', save_best_only=True,
                                                        save_weights_only=True, monitor='val_accuracy', mode='max')
        print(f'Training model {model_name}...')
        history = model.fit((rna_one_hot_X_train, protein_one_hot_X_train, rna_encoding_train, pro_encoding_train),
                            y_train, epochs=epochs, batch_size=batch_size, validation_data=(
                (rna_one_hot_X_val, protein_one_hot_X_val, rna_encoding_val, pro_encoding_val), y_val),
                            callbacks=[checkpoint])
        model.load_weights(f'models/{model_name + suffix}.h5')
        if not os.path.isfile(csv_path):
            with open(csv_path, 'w') as f:
                f.write('Dataset,Model,accuracy,roc_score,recall,precision,f1,specificity,ppv,prc_score\n')
        results = {}
        y_pred = model.predict((rna_one_hot_X_val, protein_one_hot_X_val, rna_encoding_val, pro_encoding_val))
        pred_test = [1 if i > 0.5 else 0 for i in y_pred]
        results['accuracy'] = accuracy_score(y_val, pred_test)
        results['roc_score'] = roc_auc_score(y_val, y_pred)
        results['recall'] = recall_score(y_val, pred_test)
        results['precision'] = precision_score(y_val, pred_test)
        results['f1'] = f1_score(y_val, pred_test)
        tn, fp, fn, tp = confusion_matrix(y_val, pred_test).ravel()
        results['specificity'] = tn / (tn + fp)
        results['ppv'] = tp / (tp + fp)
        results['prc_score'] = average_precision_score(y_val, y_pred)
        results["mcc"] = matthews_corrcoef(y_val, y_pred)
        print(results)
        with open(csv_path, 'a') as f:
            f.write(
                f"{dataset},{model_name},{results['accuracy']},{results['roc_score']},{results['recall']},{results['precision']},{results['f1']},{results['specificity']},{results['ppv']},{results['prc_score']}\n")


if __name__ == "__main__":
    data_path = "./data/RPI18072/pretrain_data.csv"
    # data_path = "./Data/2241.csv"
    lr = 0.001
    max_length_pro = 1000
    max_length_rna = 2500
    epochs = 100
    # max_len = 1000
    batch_size = 256
    np.random.seed(42)
    set_seed(42)
    # data = pd.read_csv(data_path, sep="\t")
    data = pd.read_csv(data_path)
    data = data.sample(frac=1).reset_index(drop=True)

    SIZE = 18072
    data = data[:SIZE]

    pro_encoding_array = Protein_CTDD(data)
    print("Protein_CTDD Finished")
    rna_encoding_array = Z_curve_144bit(data)
    print("Z_curve_144bit Finished")
    (
        rna_one_hot_X,
        protein_one_hot_X,
        y,
        rna_vocab_size,
        protein_vocab_size,
        rna_vocab_to_int,
        protein_vocab_to_int,
    ) = encode_data(data)
    del data
    gc.collect()
    print("encode_data Finished")

    rna_one_hot_X = np.transpose(rna_one_hot_X, [1, 0, 3, 2])
    rna_one_hot_X = rna_one_hot_X.squeeze(1)
    protein_one_hot_X = np.transpose(protein_one_hot_X, [1, 0, 3, 2])
    protein_one_hot_X = protein_one_hot_X.squeeze(1)
    split_num = int(SIZE * 0.8)
    X_train_rna, X_train_pro = rna_encoding_array[1:split_num, 2:].astype(
        np.float16
    ), pro_encoding_array[1:split_num, 2:].astype(np.float16)
    print("X_train get")
    X_val_rna, X_val_pro = rna_encoding_array[split_num:, 2:].astype(
        np.float16
    ), pro_encoding_array[split_num:, 2:].astype(np.float16)
    del pro_encoding_array, rna_encoding_array
    gc.collect()
    print("X_val get并回收pro_encoding_array, rna_encoding_array")

    rna_one_hot_X_train, protein_one_hot_X_train = rna_one_hot_X[
                                                   : split_num - 1
                                                   ].astype(np.float16), protein_one_hot_X[: split_num - 1].astype(np.float16)
    rna_one_hot_X_val, protein_one_hot_X_val = rna_one_hot_X[split_num - 1:].astype(
        np.float16
    ), protein_one_hot_X[split_num - 1:].astype(np.float16)
    del rna_one_hot_X, protein_one_hot_X
    gc.collect()
    print("回收了rna_one_hot_X, protein_one_hot_X")

    y_train, y_val = y[: split_num - 1].astype(np.float16), y[split_num - 1:].astype(np.float16)
    del y
    gc.collect()
    print("回收了y")

    X_train_rna, X_train_pro = np.array(X_train_rna), np.array(X_train_pro)
    X_val_rna, X_val_pro = np.array(X_val_rna), np.array(X_val_pro)
    y_train, y_val = np.array(y_train), np.array(y_val)

    rna_encoding_train, pro_encoding_train = X_train_rna, X_train_pro
    del X_train_rna, X_train_pro
    gc.collect()
    print("回收了X_train_rna, X_train_pro")
    rna_encoding_val, pro_encoding_val = X_val_rna, X_val_pro
    del X_val_rna, X_val_pro
    gc.collect()
    print("回收了X_val_rna, X_val_pro")

    rna_one_hot_X_train, protein_one_hot_X_train = rna_one_hot_X_train[
                                                   0:split_num
                                                   ].astype(np.float16), protein_one_hot_X_train[0:split_num].astype(np.float16)
    rna_one_hot_X_val, protein_one_hot_X_val = rna_one_hot_X_val[0:split_num].astype(
        np.float16
    ), protein_one_hot_X_val[0:split_num].astype(np.float16)

    print(rna_one_hot_X_train.shape, protein_one_hot_X_train.shape, rna_encoding_train.shape, pro_encoding_train.shape)
    train_all_models(
        rna_one_hot_X_train,
        protein_one_hot_X_train,
        rna_encoding_train,
        pro_encoding_train,
        rna_one_hot_X_val,
        protein_one_hot_X_val,
        rna_encoding_val,
        pro_encoding_val,
        y_train,
        y_val,
        max_length_rna,
        max_length_pro,
        rna_vocab_size,
        protein_vocab_size,
        lr,
        epochs
    )

    print("Model training finished.")
    for model_name in ["CNN", "lstm", "rnn", "bilstm", "CBIL–VHPLI"]:
        model = define_model(model_name, max_length_rna, rna_vocab_size, protein_vocab_size)
        model.load_weights(f"models/{model_name}pretrained.h5")
        y_pred = model.predict(
            (
                rna_one_hot_X_val,
                protein_one_hot_X_val,
                rna_encoding_val,
                pro_encoding_val,
            )
        )
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        auc_score = roc_auc_score(y_val, y_pred)

        plt.plot(fpr, tpr, label=f"{model_name.upper()} AUC={auc_score:.4f}")
    plt.legend()
    plt.plot([0, 1], [0, 1], "--")
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    plt.savefig(fname="./figs/ROC_pretrained.png")
