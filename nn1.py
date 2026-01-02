import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def load_file():
    from pathlib import Path
    project_root= Path.cwd()
    filepath = str(project_root)+"/data/scaled/pr_water_rho_scaled.csv"
    #print(filepath)
    rho_orig = pd.read_csv(filepath)
    #print(rho_orig.head())
    return rho_orig


def split_data(rho):

    x = rho[["T_K","P_Pa"]]
    y = rho[["rho_liquid", "rho_vapor"]]    
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
    #print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    
    return x_train,x_test,y_train,y_test


#print(xt_train.shape,xt_test.shape,yt_train.shape,yt_test.shape)

tf.random.set_seed(42)


def build_rho_model(
    n_hidden=[100, 50, 20],
    activation=["relu","relu","relu"],
    lr=1e-4
):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,)))

    for n in range(len(n_hidden)):
        model.add(tf.keras.layers.Dense(n_hidden[n], activation=activation[n]))

    model.add(tf.keras.layers.Dense(2))  # rho_liq, rho_vap

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.mape,
        metrics=["mape"]
    )
    return model

"""
model1 = build_rho_model()


print(model1.summary)

h1 = model1.fit(x_train, y_train,
                    epochs=1000,
                    batch_size = 32,
                    validation_data=(x_test, y_test))
"""

def plot_loss(h):
    loss = h.history['loss']
    val_loss = h.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    #plt.xlim(0.000, 1000)
    #plt.ylim(0.0, 1000)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

