import os
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from unet import UNet
import matplotlib.pyplot as plt
import time
import datetime as dt
from tkinter import *
import tkinter.filedialog
import subprocess
import cv2
import glob

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


def train(folder, batch_size, num_epoch):
    project_folder = os.getcwd() + os.sep + folder
    X_train, file_names = load_X(project_folder + os.sep + "2_tarin_images")
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
    model.save_weights(filename)

    plt.plot(history.history["loss"])
    plt.title(f"Training loss : batch_size={batch_size}, epochs={num_epoch}")
    plt.legend()
    filename = f"loss_batch={batch_size}_epochs={num_epoch}.png"
    plt.savefig(filename)






# インプット画像を読み込む関数
def load_X(path):
    image_files = os.listdir(path)                                        #フォルダ内ファイル名リストを取得
    image_files.sort()                                                           #ファイル名昇順でソート
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32) #全て0の配列を生成
    for i, image_file in enumerate(image_files):
        image = cv2.imread(path + os.sep + image_file)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        images[i] = normalize_x(image)
    return images, image_files
 
 
# ラベル画像を読み込む関数
def load_Y(path):
    import os, cv2
 
    image_files = os.listdir(path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(path + os.sep + image_file)#入力画像をNumPy配列ndarrayとして読み込み
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))                         #IMAGE_SIZE×IMAGE_SIZEの大きさにリサイズ
        #image = image[:, :, np.newaxis]
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
 
# 推論結果のフォルダを開く関数
def open_folder():
    path = r'C:\ml\ML_doujou\14_u-net_keras\predictions'
    subprocess.Popen(['explorer', path], shell=True)
 
# 学習＋推論の場合：False　推論のみの場合：True
##predictiononly=True
 
# U-Netのトレーニングを実行する関数
##if predictiononly==False:
    ##print('学習＋推論実施します')
def train_unet():
    button_1a['text'] = '学習中です'
    # trainingDataフォルダ配下にleft_imagesフォルダを置いている
    X_train, file_names = load_X('trainingData' + os.sep + 'left_images')
    # trainingDataフォルダ配下にleft_groundTruthフォルダを置いている
    Y_train = load_Y('trainingData' + os.sep + 'left_groundTruth')
 
    # 入力はRGB3チャンネル
    input_channel_count = 3
    # 出力はグレースケール1チャンネル
    output_channel_count = 3
    # 一番初めのConvolutionフィルタ枚数は64
    first_layer_filter_count = 64
    # U-Netの生成
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.compile(loss=dice_coef_loss, optimizer=Adam(), metrics=[dice_coef])
 
    BATCH_SIZE = 8
    a = txt.get()
    NUM_EPOCH = int (a)
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1)
    model.save_weights('unet_weights.hdf5_100epoch')
 
    # Loss値のグラフ表示
 
    epochs = NUM_EPOCH
 
    plt.plot(history.history['loss'])
    plt.title('Training loss')
    plt.legend()
    plt.savefig('loss.png')
    button_1a['text'] = '学習完了しました'
##else:
##    print('推論のみ実施します')
 
 
# 学習後のU-Netによる予測を行う関数
def predict():
    
    target_file = tkinter.filedialog.askopenfilename(title="推論に使用する学習モデルを選択してください")
    button_2a['text'] = '推論中です'
    
    import cv2
 
    start_time = time.perf_counter()     
                                       
    X_test, file_names = load_X('testData' + os.sep + 'left_images')
    # testDataフォルダ配下にleft_imagesフォルダを置いている
    ##start_time1 = time.time()
    start_time1 = time.perf_counter()                                            #読込時間計測
    X_test, file_names = load_X('testData' + os.sep + 'left_images')
    ##end_time1 = time.time()
    end_time1 = time.perf_counter()
    print(f"elapsed time1: {end_time1-start_time1:.3f}sec")
    
    #start_time2 = time.time()
    start_time2 = time.perf_counter()                                            #推論時間計測
    input_channel_count = 3
    output_channel_count = 3
    first_layer_filter_count = 64
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.load_weights(target_file)                                             #選択した学習モデル読込
    BATCH_SIZE = 8
    Y_pred = model.predict(X_test, BATCH_SIZE)
    #end_time2 = time.time()
    end_time2 = time.perf_counter()
    print(f"elapsed time2: {end_time2-start_time2:.3f}sec")
    
    #start_time3 = time.time()
    start_time3 = time.perf_counter()                                            #出力時間計測
    dt_now = dt.datetime.now()
    yyyymmddHHMM = dt_now.strftime('%Y%m%d%H%M')
    directory_name = 'prediction_'+yyyymmddHHMM                                 #推論結果出力フォルダ名
    os.mkdir('predictions' + os.sep + directory_name)
    for i, y in enumerate(Y_pred):
        # testDataフォルダ配下にleft_imagesフォルダを置いている
        img = cv2.imread('testData' + os.sep + 'left_images' + os.sep + file_names[i])
        y = cv2.resize(y, (img.shape[1], img.shape[0]))
        ##zero_i="{0:03d}".format(i)
        cv2.imwrite('predictions' + os.sep + directory_name + os.sep+'pred_' + file_names[i], denormalize_y(y))
    #end_time3 = time.time()
    end_time3 = time.perf_counter()
    print(f"elapsed time3: {end_time3-start_time3:.3f}sec")
        
    end_time = time.perf_counter()
    print(f"elapsed time: {end_time-start_time:.3f}sec")
    
    button_2a['text'] = '推論完了しました'
    open_folder()
 
def main():
    # GUI
    root = Tk()
    root.title('U-net学習＆推論')   # 画面タイトル設定
    root.geometry('1000x500')       # 画面サイズ設定
    root.resizable(False, False)
    button_1a = Button(root, width=20, height=5, font=5, text='学習', command=train_unet)
    button_1a.pack(padx=5, pady=10)
    button_2a = Button(root, width=20, height=5, font=5, text='推論', command=predict)
    button_2a.pack(padx=5, pady=10)
    
    lbl = tkinter.Label(text='学習エポック数') # ラベル
    lbl.place(x=30, y=70)       
    
    txt = tkinter.Entry(width=20)   # エポック数入力ボックス
    txt.place(x=150, y=70)
    
    root.mainloop()
