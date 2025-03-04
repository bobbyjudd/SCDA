import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot as plt

np.random.seed(seed=28213)
for rare_com in ['common', 'rare']:
    for dim in [228, 2500]:
        input_name = 'data/{}_train_{}.tsv'.format(rare_com, dim)#'data/yeast_genotype_train.txt'
        #df_ori = pd.read_csv(input_name, index_col=0).drop(columns=['IID','PAT','MAT','SEX','PHENOTYPE'])+1
        df_ori = pd.read_csv(input_name, sep='\t', index_col=0)

        # Preprocessing
        df_onehot = to_categorical(df_ori)

        # split df to train and valid
        train_X, valid_X = train_test_split(df_onehot, test_size=0.2)

        # hyperparameters
        for missing_perc in [0.05, 0.1, 0.2, 0.3, 0.4]:
            if rare_com == 'mid' and dim == 2500:
                continue
            print('Starting {} | {} | {}'.format(rare_com, dim, missing_perc))
            # training
            batch_size = 32
            lr = 1e-3
            epochs = 1000

            # conv1D
            feature_size = train_X.shape[1]
            inChannel = train_X.shape[2]
            kr = 1e-4
            drop_prec = 0.25


            SCDA = Sequential()
            # encoder
            SCDA.add(Conv1D(32, 5, padding='same',activation='relu',kernel_regularizer=l1(kr),input_shape=(feature_size, inChannel)))
            SCDA.add(MaxPooling1D(pool_size=2))
            SCDA.add(Dropout(drop_prec))
                    
            SCDA.add(Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=l1(kr))) 
            SCDA.add(MaxPooling1D(pool_size=2)) 
            SCDA.add(Dropout(drop_prec))

            # bridge
            SCDA.add(Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l1(kr)))

            # decoder
            SCDA.add(Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=l1(kr))) 
            SCDA.add(UpSampling1D(2)) 
            SCDA.add(Dropout(drop_prec))
                    
            SCDA.add(Conv1D(32, 5, padding='same', activation='relu', kernel_regularizer=l1(kr))) 
            SCDA.add(UpSampling1D(2))
            SCDA.add(Dropout(drop_prec))

            SCDA.add(Conv1D(inChannel, 5, activation='softmax', padding='same')) 


            # compile
            SCDA.compile(loss='categorical_crossentropy', 
                                optimizer='adam',
                                metrics=['accuracy'])

            # TODO Custom loss function

            SCDA.summary()

            # Generates data for denoising autoencoder.
            class DataGenerator(keras.utils.Sequence):
                def __init__(self, batch_size, x_dataset, missing_perc=0.1, shuffle=True):
                    self.batch_size = batch_size
                    self.x = x_dataset
                    self.missing_perc = missing_perc
                    self.shuffle = shuffle
                    # triggered once at the very beginning as well as at the end of each epoch.
                    self.on_epoch_end()

                def __len__(self):
                    # Denote the number of batches per epoch
                    return int(np.floor(self.x.shape[0] / self.batch_size))

                def __getitem__(self, index):
                    # Generates one batch of data
                    indexes = self.indexes[index * self.batch_size:(
                        index + 1) * self.batch_size]
                    self.x_missing = self.x[indexes].copy()

                    # Generates missing genotypes
                    # different missing loci for each individuals
                    for i in range(self.x_missing.shape[0]):
                        missing_size = int(self.missing_perc * self.x_missing.shape[1])
                        missing_index = np.random.randint(
                            self.x_missing.shape[1], size=missing_size)
                        # missing loci are encoded as [0, 0]
                        self.x_missing[i, missing_index, :] = [1] + [0]*(df_onehot.shape[2]-1)

                    return self.x_missing, self.x[indexes]

                def on_epoch_end(self):
                    # Update indexes after each epoch
                    self.indexes = np.arange(self.x.shape[0])
                    if self.shuffle == True:
                        np.random.shuffle(self.indexes)

            train_generator = DataGenerator(
                batch_size=batch_size, x_dataset=train_X, missing_perc=missing_perc)
            valid_generator = DataGenerator(
                batch_size=batch_size, x_dataset=valid_X, missing_perc=missing_perc)


            # Training

            # early stopping call back with val_loss monitor
            EarlyStopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=10,
                verbose=0,
                mode='auto',
                baseline=None,
                restore_best_weights=True)

            # model checkpoint call back with val_acc monitor
            ModelCheckpoint = keras.callbacks.ModelCheckpoint(
                'model/SCDA_checkpoint.{epoch:02d}.h5',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                save_freq='epoch')

            SCDA_train = SCDA.fit(
                x=train_generator,
                validation_data=valid_generator,
                epochs=epochs,
                verbose=1,
                callbacks=[EarlyStopping]
            )

            # plot loss curve on validation data
            loss = SCDA_train.history['loss']
            val_loss = SCDA_train.history['val_loss']

            plt.figure()
            plt.plot(range(len(loss)), loss, 'bo', label='Training loss')
            plt.plot(range(len(val_loss)), val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss {} | Dim {} | {}% Missing'.format(rare_com, dim, int(missing_perc*100)))
            plt.legend()
            plt.savefig('plots/loss_{}_{}_{}'.format(rare_com, dim, int(missing_perc*100)))

            # plot accuracy curve on validation data
            acc = SCDA_train.history['accuracy']
            val_acc = SCDA_train.history['val_accuracy']
            plt.figure()
            plt.plot(range(len(acc)), acc, 'bo', label='Training acc')
            plt.plot(range(len(val_acc)), val_acc, 'b', label='Validation acc')
            plt.title('Training and validation acc {} | Dim {} | {}% Missing'.format(rare_com, dim, int(missing_perc*100)))
            plt.legend()
            plt.savefig('plots/acc_{}_{}_{}'.format(rare_com, dim, int(missing_perc*100)))

            # # Save model
            SCDA.save('model/SCDA_{}_{}_{}.h5'.format(rare_com, dim, int(missing_perc*100)))  # creates a HDF5 file 'SCDA.h5'