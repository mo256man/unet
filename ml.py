import os
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from unet import UNet
import matplotlib.pyplot as plt
import time
import datetime as dt
import cv2
import glob
import math

IMAGE_SIZE = 256

# 値を-1から1に正規化する関数
def normalize_x(image):
    image = image/127.5 - 1
    return image

# 値を0から1に正規化する関数
def normalize_y(image):
    image = image/255
    return image


# 値を0から255に戻す関数
def denormalize_y(image):
    image = image*255
    return image


def train(project_folder, batch_size, num_epoch):
    X_train, _ = load_X(project_folder + os.sep + "2_train_images")
    Y_train = load_Y(project_folder + os.sep + "2_train_labels")

    input_channel_count = 3         # 入力チャンネル数（RGB 3チャンネル）
    output_channel_count = 3        # 出力チャンネル数（RGB 3チャンネル）
    first_layer_filter_count = 64   # 一番初めのConvolutionフィルタ枚数は64

    # U-Netの生成
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.compile(loss=dice_coef_loss, optimizer=Adam(), metrics=[dice_coef])
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epoch, verbose=1)
    filename = f"weights_batch={batch_size}_epochs={num_epoch}.hdf5"
    model.save_weights(project_folder + os.sep + filename)

    plt.plot(history.history["loss"])
    plt.title(f"Training loss : batch_size={batch_size}, epochs={num_epoch}")
    plt.legend()
    filename = f"loss_batch={batch_size}_epochs={num_epoch}.png"
    plt.savefig(project_folder + os.sep + filename)


def predict(project_folder, modelfile):
    current_path = os.getcwd()
    start_time = time.perf_counter()    

    start_time1 = time.perf_counter()                                            #読込時間計測
    X_test, filenames = load_X(project_folder + "4_test_images")
    end_time1 = time.perf_counter()
    print(f"elapsed time1: {end_time1-start_time1:.3f}sec")
    
    start_time2 = time.perf_counter()                                            #推論時間計測
    input_channel_count = 3
    output_channel_count = 3
    first_layer_filter_count = 64
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.load_weights(modelfile)                                             #選択した学習モデル読込
    BATCH_SIZE = 8
    Y_pred = model.predict(X_test, BATCH_SIZE)
    end_time2 = time.perf_counter()
    print(f"elapsed time2: {end_time2-start_time2:.3f}sec")
    
    start_time3 = time.perf_counter()                                            #出力時間計測
    dt_now = dt.datetime.now()
    yyyymmddHHMM = dt_now.strftime('%Y%m%d%H%M')
    directory_name = 'prediction_'+yyyymmddHHMM                                     #推論結果出力フォルダ名
    # os.mkdir('predictions' + os.sep + directory_name)

    for y, filename in zip(Y_pred, filenames):
        img = cv2.imread(project_folder + "4_test_images" + os.sep + filename)
        y = cv2.resize(y, (img.shape[1], img.shape[0]))
        pred_filename = "pred," + filename
        pred_image = denormalize_y(y)
        cv2.imwrite(project_folder + os.sep + "5_predict_images" + os.sep + pred_filename, pred_image)

    end_time3 = time.perf_counter()
    print(f"elapsed time3: {end_time3-start_time3:.3f}sec")
        
    end_time = time.perf_counter()
    print(f"elapsed time: {end_time-start_time:.3f}sec")

# インプット画像を読み込む関数
def load_X(path):
    image_files = os.listdir(path)                                                  #フォルダ内ファイル名リストを取得
    image_files.sort()                                                              #ファイル名昇順でソート
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)    #全て0の配列を生成
    filenames = []
    for i, image_file in enumerate(image_files):
        image = cv2.imread(path + os.sep + image_file)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        images[i] = normalize_x(image)
        filename = os.path.basename(image_file)                 # フルパスからファイル名を取得
        filenames.append(filename)
    
    return images, filenames



# インプット画像を読み込む関数
def load_X_0(path):
    image_files = os.listdir(path)                                                  #フォルダ内ファイル名リストを取得
    image_files.sort()                                                              #ファイル名昇順でソート
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)    #全て0の配列を生成
    for i, image_file in enumerate(image_files):
        image = cv2.imread(path + os.sep + image_file)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        images[i] = normalize_x(image)
    return images, image_files


# ラベル画像を読み込む関数
def load_Y(path):
    image_files = os.listdir(path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(path + os.sep + image_file)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))                         #IMAGE_SIZE×IMAGE_SIZEの大きさにリサイズ
        images[i] = normalize_y(image)
    return images



# ダイス係数を計算する関数　類似度を数値化。大きいほど似ている　正解画像と推論結果がどれだけ近いか？
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)
 
 
# ロス関数　ロス関数＝１－ダイス係数
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

if __name__ == "__main__":
    print("このプログラムは単独では動きません")
