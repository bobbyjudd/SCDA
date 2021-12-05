import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

from fancyimpute import KNN, SoftImpute, IterativeSVD
import matplotlib.pyplot as plt

impute_methods = {
    'KNN':KNN,
    'SoftImpute':SoftImpute,
    'IterativeSVD':IterativeSVD
    }
mps = [0.05, 0.1, 0.2, 0.3, 0.4]
for freq in ['rare']:#mid', 'common', 'rare']:
    plot_map = {'SCDA':[], 'KNN':[], 'SoftImpute':[], 'IterativeSVD':[]}
    for missing_perc in mps:
        for dim in [228, 2500]:
            SCDA = load_model('model/SCDA_{}_{}_{}.h5'.format(freq, dim, int(missing_perc*100)))

            # ## Loading data
            input_name = 'data/{}_test_{}.tsv'.format(freq, dim)
            df_ori = pd.read_csv(input_name, sep='\t', index_col=0)

            with open('data/{}_af_list_{}.txt'.format(freq, dim)) as f:
                af_labels = f.read().split(',')

            test_X = df_ori.to_numpy(dtype=float)
            test_X_oh = to_categorical(df_ori)
            test_X_missing = df_ori.to_numpy(dtype=float)
            test_X_missing_oh = test_X_oh.copy()

            def cal_prob(predict_missing_onehot, n=None):
                if n is None:
                    n = predict_missing_onehot.shape[2]
                # calcaulate the probility of genotype 0, 1, 2
                predict_prob = predict_missing_onehot[:,:,1:n] / predict_missing_onehot[:,:,1:n].sum(axis=2, keepdims=True)
                return predict_prob[0]

            print('Generating missing values and eval SCDA model')
            avg_accuracy = []
            missing_inds = []
            correct_snps = np.array([0]*len(af_labels))
            total_snps = np.array([0]*len(af_labels))
            for i in range(test_X_missing.shape[0]):
                # Generates missing genotypes
                missing_size = int(missing_perc * test_X_missing.shape[1])
                missing_index = np.random.randint(test_X_missing.shape[1],
                                                size=missing_size)
                test_X_missing_oh[i, missing_index, :] = [1] + [0]*(test_X_oh.shape[2]-1)
                test_X_missing[i, missing_index] = np.nan   # Mark missing for legacy impute methods
                missing_inds.append(missing_index)

                # predict
                predict_onehot = SCDA.predict(test_X_missing_oh[i:i + 1, :, :])

                # only care the missing position
                predict_missing_onehot = predict_onehot[0:1, missing_index, :]

                # calculate probability and save file.
                predict_prob = cal_prob(predict_missing_onehot)
                pd.DataFrame(predict_prob).to_csv('results/{}.csv'.format(df_ori.index[i]),
                                                header=list(range(1,predict_missing_onehot.shape[2])),
                                                index=False)
                # predict label
                predict_missing = np.argmax(predict_missing_onehot, axis=2)
                # real label
                label_missing_onehot = test_X_oh[i:i + 1, missing_index, :]
                label_missing = np.argmax(label_missing_onehot, axis=2)

                # accuracy
                correct_prediction = np.equal(predict_missing, label_missing)
                for j, snp_index in enumerate(missing_index):
                    correct_snps[snp_index] += 1 if correct_prediction[0][j] else 0
                    total_snps[snp_index] += 1
                accuracy = np.mean(correct_prediction)
                print('[SCDA]: {}/{}, sample ID: {}, accuracy: {:.4f}'.format(
                    i, test_X_missing.shape[0], df_ori.index[i], accuracy))

                avg_accuracy.append(accuracy)

            plot_map['SCDA'].append(np.mean(avg_accuracy))
            print('The average imputation accuracy with {} missing genotypes is {:.4f}: '
                .format(missing_perc, plot_map['SCDA'][-1]))

            box_map = {}
            for i in range(len(correct_snps)):
                if total_snps[i] != 0:
                    if float(af_labels[i]) not in box_map:
                        box_map[float(af_labels[i])] = []
                    box_map[float(af_labels[i])].append(correct_snps[i]/total_snps[i])

            positions = sorted(box_map.keys())
            plot_data = [box_map[positions[k]] for k in range(len(positions))]
            if len(positions) == 1:
                plot_min, plot_max = min(box_map[positions[0]]), max(box_map[positions[0]])
                plt.hist(box_map[positions[0]], bins=np.linspace(plot_min, plot_max,1+int(100*(plot_max-plot_min))))
                plt.title('SCDA | {} | {}% Missing'.format(freq,int(missing_perc*100)))
                plt.ylabel('Count')
                plt.xlabel('Accuracy')
            else:
                plt.title('SCDA | {} | {}% Missing'.format(freq,int(missing_perc*100)))
                plt.xlabel('Allele Frequency')
                plt.ylabel('Accuracy')
                plt.boxplot(plot_data, labels=[str(positions[p]) for p in range(len(positions))])
            plt.savefig('plots/SCDA_{}_{}_{}'.format(freq, dim,int(missing_perc*100)))
            plt.clf()


            for name,m in impute_methods.items():
                print('Predicting missing using {}'.format(name))
                if name == 'KNN':
                    legacy_preds = m(k=3).fit_transform(test_X_missing)
                else:
                    legacy_preds = m().fit_transform(test_X_missing)
                legacy_preds = np.around(legacy_preds)

                # accuracy
                correct_snps = np.array([0]*len(af_labels))
                total_snps = np.array([0]*len(af_labels))
                correct_prediction = []
                for i in range(legacy_preds.shape[0]):   
                    current_cp = np.equal(legacy_preds[i:i + 1, missing_inds[i]], test_X[i:i + 1, missing_inds[i]])
                    correct_prediction.append(current_cp)
                    for j, snp_index in enumerate(missing_inds[i]):
                        correct_snps[snp_index] += 1 if current_cp[0][j] else 0
                        total_snps[snp_index] += 1
                
                accuracy = np.mean(correct_prediction)
                print('{} full test set accuracy: {:.4f}'.format(name, accuracy))
                plot_map[name].append(accuracy)

                box_map = {}
                for i in range(len(correct_snps)):
                    if total_snps[i] != 0:
                        if float(af_labels[i]) not in box_map:
                            box_map[float(af_labels[i])] = []
                        box_map[float(af_labels[i])].append(correct_snps[i]/total_snps[i])
                
                positions = sorted(box_map.keys())
                plot_data = [box_map[positions[k]] for k in range(len(positions))]
                if len(positions) == 1:
                    plot_min, plot_max = min(box_map[positions[0]]), max(box_map[positions[0]])
                    plt.hist(box_map[positions[0]], bins=np.linspace(plot_min, plot_max,1+int(100*(plot_max-plot_min))))
                    plt.title('{} | {} | {}% Missing'.format(name, freq,int(missing_perc*100)))
                    plt.ylabel('Count')
                    plt.xlabel('Accuracy')
                else:
                    plt.title('{} | {} | {}% Missing'.format(name, freq,int(missing_perc*100)))
                    plt.xlabel('Allele Frequency')
                    plt.ylabel('Accuracy')
                    plt.boxplot(plot_data, labels=[str(positions[p]) for p in range(len(positions))])
                plt.savefig('plots/{}_{}_{}_{}'.format(name, freq, dim, int(missing_perc*100)))
                plt.clf()
    
    for name in impute_methods.keys():
        plt.plot([int(p*100) for p in mps], plot_map[name], label=name)
    plt.plot([int(p*100) for p in mps], plot_map['SCDA'], label='SCDA')
    plt.title('Accuracy for {} SNPs'.format(freq))
    plt.xlabel('% Missing')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('plots/all_{}_{}_acc'.format(freq, dim))
    plt.clf()
