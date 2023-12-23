import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import Model
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.layers import Dense, Dropout,LeakyReLU, Input
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers

def train_autoencoder(dim_features, dim_latent, dim_layer2, x_train_data, x_train_data_noisy,n_batch=20, optimizer1='adam',loss1='mse',n_epochs=200,seed_value=0):
    m_w_prime_init = initializers.glorot_uniform(seed=seed_value)
    m_w_prime_inter = initializers.glorot_uniform(seed=seed_value)
    m_w_init = initializers.glorot_uniform(seed=seed_value)
    m_w_inter = initializers.glorot_uniform(seed=seed_value)
    input_dim = Input(shape=(dim_features,))

    encoded = Dense(dim_layer2, bias_initializer='zeros', kernel_initializer=m_w_inter)(input_dim)

    encoded = layers.LeakyReLU(alpha=0.3)(encoded)

    encoded = Dense(dim_latent,
                    bias_initializer='zeros',
                    kernel_initializer=m_w_init)(encoded)

    encoded = layers.LeakyReLU(alpha=0.3)(encoded)

    # Add new layer
    decoded = Dense(dim_layer2, bias_initializer='zeros', kernel_initializer=m_w_prime_inter)(encoded)
    decoded = layers.LeakyReLU(alpha=0.3)(decoded)

    decoded = Dense(dim_features,
                    activation='sigmoid',
                    bias_initializer='zeros',
                    kernel_initializer=m_w_prime_init)(decoded)

    autoencoder = Model(input_dim, decoded)
    encoder = Model(input_dim, encoded)

    autoencoder.summary()
    autoencoder.compile(optimizer=optimizer1,
                        loss=loss1)

    ae_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)

    history = autoencoder.fit(x_train_data_noisy,
                              x_train_data,
                              epochs=n_epochs,
                            callbacks=[ae_earlystop],
                              batch_size=int(len(x_train_data)/n_batch),
                              shuffle=True,
                              verbose=1,
                              validation_split=0.2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return autoencoder, encoder, history
def SAE_dim_reduction(train,test,Emb_size=10,loss='mse',optimizer='adam',epochs= 100,n_batches=40):

    train= np.asarray(train)
    test= np.asarray(test)

    def scale_datasets(train,test):

        standard_scaler = StandardScaler()
        x_train_scaled = pd.DataFrame(
          standard_scaler.fit_transform(train),
          columns=np.arange(0,len(train[0])))

        x_test_scaled = pd.DataFrame(
       standard_scaler.transform(test),
       columns=np.arange(0,len(test[0])))
        return x_train_scaled, x_test_scaled


    train1,test1=scale_datasets(train,test)


    autoencoder, encoder, history = train_autoencoder(len(train[0]), Emb_size, 300, train1, train1,n_batch=n_batches, optimizer1=optimizer,loss1=loss,n_epochs=epochs)


    input2=pd.concat([pd.DataFrame(autoencoder.predict(train1)),train1])


    autoencoder, encoder, history = train_autoencoder(len(train[0]), Emb_size, 300, input2, input2,n_batch=n_batches, optimizer1=optimizer,loss1=loss,n_epochs=epochs)

    input3=pd.concat([pd.DataFrame(autoencoder.predict(input2)),input2])


    autoencoder, encoder, history = train_autoencoder(len(train[0]), Emb_size, 300, input3, input3,n_batch=n_batches, optimizer1=optimizer,loss1=loss,n_epochs=epochs)


    emb_train=encoder.predict(train1)
    emb_test=encoder.predict(test1)
    mae_train=mean_squared_error(train1,autoencoder.predict(train1))
    mae_test=mean_squared_error(test1,autoencoder.predict(test1))
    print('mae train: ',mae_train)
    print('mae test:',mae_test)
    return emb_train,emb_test,mae_train,mae_test


def AE_dim_reduction(train, test, Emb_size=10, loss='mse', optimizer='adam', epochs=100, n_batches=40):
    train = np.asarray(train)
    test = np.asarray(test)

    def scale_datasets(train, test):
        standard_scaler = StandardScaler()
        x_train_scaled = pd.DataFrame(
            standard_scaler.fit_transform(train),
            columns=np.arange(0, len(train[0])))

        x_test_scaled = pd.DataFrame(
            standard_scaler.transform(test),
            columns=np.arange(0, len(test[0])))
        return x_train_scaled, x_test_scaled

    train1, test1 = scale_datasets(train, test)

    autoencoder, encoder, history = train_autoencoder(len(train[0]), Emb_size, 300, train1, train1, n_batch=n_batches,
                                                      optimizer1=optimizer, loss1=loss, n_epochs=epochs)
    emb_train = encoder.predict(train1)
    emb_test = encoder.predict(test1)
    mae_train = mean_squared_error(train1, autoencoder.predict(train1))
    mae_test = mean_squared_error(test1, autoencoder.predict(test1))
    print('mae train: ', mae_train)
    print('mae test:', mae_test)
    return emb_train, emb_test, mae_train, mae_test