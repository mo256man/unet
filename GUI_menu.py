import PySimpleGUI as sg
import numpy as np
import cv2
import glob
import os
import math
from ml import *

IMAGE_SIZE = 256

class Gui():
    def __init__(self):
        self.is_folder_ok = False                       # フォルダ内に必要なサブフォルダが揃っているかどうか　の初期値
        self.hdf5_files = []                            # hdf5ファイルを取得する
        self.project_folders = glob.glob("./**/")       # サブフォルダを取得する

        layout_folder = [[sg.Text("新規作成", size=(10,1))],
                         [sg.InputText("", size=(40,1), key="-FOLDER0-"),
                          sg.Button("create", key="-FOLDER_CREATE-")],
                          [sg.Text("選択")],
                         [sg.Combo(self.project_folders, default_value="", size=(40,1), key="-FOLDER-", readonly=True),
                          sg.Button("check", key="-FOLDER_CHECK-"),
                          sg.Text("✕", key="-RESULT_FOLDER-")]]
        frame_folder = sg.Frame("プロジェクトフォルダー", layout_folder)
        layout_img = [[sg.Text("", size=(30,1))]]
#        layout_img = [[sg.Text("size", size=(10,1)),
#                       sg.InputText("256", size=(10,1), justification="right", key="-SIZE-")]]
        btn_img = sg.Button("START", key="-IMG_START-", size=(12,1))
        frame_img = sg.Frame("データ拡張", [[sg.Column(layout_img), btn_img]])

        layout_teach = [[sg.Text("batch size", size=(10,1)),
                       sg.InputText("8", size=(10,1), justification="right", key="-BATCH-")],
                       [sg.Text("epoch", size=(10,1)),
                        sg.InputText("30", size=(10,1), justification="right", key="-EPOCH-")]]
        btn_teach = sg.Button("START", key="-TRAIN_START-", size=(12,3))
        frame_teach = sg.Frame("学習", [[sg.Column(layout_teach), btn_teach]])

        layout_predict = [[sg.Text("モデルデータ", size=(10,1))],
                          [sg.Combo(self.hdf5_files, default_value="", size=(40,1), key="-MODELS-", readonly=True)]]
        btn_predict = sg.Button("START", key="-PREDICT_START-", size=(12,3))
        frame_predict = sg.Frame("推論", [[sg.Column(layout_predict), btn_predict]])

        layout_evaluation =  [[sg.Text("3_test_labels_origin", size=(20,1))],
                              [sg.Combo([], default_value="", size=(55,1), key="-LABEL-", readonly=True)],
                              [sg.Text("6_predict_images_combine", size=(20,1))],
                              [sg.Combo([], default_value="", size=(55,1), key="-PRED-", readonly=True)],
                              [sg.Button("START", key="-EVAL_START-", size=(12,1))]
                              ]
        frame_evaluation = sg.Frame("評価", layout_evaluation)

        layout = [[sg.Text("機械学習総合アプリ", font=("Arial",20))],
                  [sg.Column([[frame_folder],[frame_img], [frame_teach],[frame_predict], [frame_evaluation]]),
                   sg.Column([[sg.Multiline("", size=(50,35),background_color="black", text_color="white", key="-ML-")]])]
                  ]

        self.window = sg.Window("UNet GUI", layout)
        

    def folder_check(self, project_folder):
        subfolders = ["1_images_origin", "1_labels_origin",
                      "2_train_images", "2_train_labels",
                      "3_test_images_origin", "3_test_labels_origin",
                      "4_test_images", "4_test_labels",
                      "5_predict_images",
                      "6_predict_images_combine"]
        current_path = os.getcwd() + os.sep + project_folder
        window.write_event_value("-UPDATE-", "")

        # フォルダの存在確認
        if project_folder=="":
            window.write_event_value("-UPDATE-", f"フォルダ名を入力してください")
        elif not os.path.isdir(current_path):
            window.write_event_value("-UPDATE-", f"{project_folder}というフォルダは存在しません")
        else:
            self.is_folder_ok = True
            myfolders = [f for f in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, f))]
            for folder in subfolders:
                if folder in myfolders:
                    files = glob.glob(current_path + os.sep + folder + os.sep + "*")
                    msg = f"◯　{folder}　{len(files)}files"
                else:
                    msg = f"✕　{folder}"
                    self.is_folder_ok = False
                window.write_event_value("-PRINT-", msg)
                
            if self.is_folder_ok:
                window.write_event_value("-FOLDER_OK-", "◯")
                self.hdf5_files = glob.glob(project_folder + os.sep + "*.hdf5")
                window["-MODELS-"].Update(values=self.hdf5_files)
                
                for folder, key in zip(["3_test_labels_origin", "6_predict_images_combine"], ["-LABEL-", "-PRED-"]):
                    files = glob.glob(project_folder + os.sep + folder + os.sep + "*")
                    window[key].Update(values=files)
                
            else:
                window.write_event_value("-FOLDER_OK-", "✕")
                window["-MODELS-"].Update(values=[])
        
    def folder_create(self, folder):
        # 存在確認
        if folder == "":
            window.write_event_value("-UPDATE-", f"サブフォルダー名を入力してください")
        elif os.path.isdir(folder):
            window.write_event_value("-UPDATE-", f"{folder}フォルダーはすでに存在します")
        else:
            window.write_event_value("-UPDATE-", f"{folder}フォルダーを作成します\n")
            os.makedirs(folder)
            subfolders = ["1_images_origin", "1_labels_origin",
                "2_train_images", "2_train_labels",
                "3_test_images_origin", "3_test_labels_origin",
                "4_test_images", "4_test_labels",
                "5_predict_images",
                "6_predict_images_combine"]
            for subfolder in subfolders:
                os.makedirs(folder + os.sep + subfolder)
            msg = "完了"
            window.write_event_value("-PRINT-", msg)
            self.project_folders = glob.glob("./**/")
            window["-FOLDER-"].update(values=self.project_folders)


def data_augumentation(folder, size):
    project_folder = os.getcwd() + os.sep + folder
    in_subfolders = ["1_images_origin", "1_labels_origin"]
    out_subfolders = ["2_train_images", "2_train_labels"]

    window.write_event_value("-UPDATE-", "データ拡張　開始\n\n")
    for out_folder in out_subfolders:
        files = glob.glob(project_folder + os.sep + out_folder + os.sep + "*")
        window.write_event_value("-PRINT-", f"{out_folder}内のファイルを全削除\n")
        for file in files:
            os.remove(file)

    for in_folder, out_folder in zip(in_subfolders, out_subfolders):
        files = glob.glob(project_folder + os.sep + in_folder + os.sep + "*")
        window.write_event_value("-PRINT-", f"{in_folder}には{len(files)}個のファイルがある")
        out_path = project_folder + os.sep + out_folder
        for file in files:
            filename = os.path.basename(file)               # フルパスからファイル名を取得
            basename, ext = os.path.splitext(filename)      # ファイル名をベースと拡張子に分ける
            img_origin = cv2.imread(file)
            height, width = img_origin.shape[:2]
            x_cnt = width // size
            y_cnt = height // size
            for i in range(x_cnt):
                x = i * size
                for j in range(y_cnt):
                    y = j * size
                    roi = img_origin[y:y+size, x:x+size]
                    filename = f"{basename}_({i}_{j}){ext}"
                    cv2.imwrite(out_path + os.sep + filename, roi)
        files = glob.glob(out_path + os.sep + "*")
        cnt = len(files)
        window.write_event_value("-PRINT-", f"　→ {cnt}個に拡張した\n")

    window.write_event_value("-PRINT-", "データ拡張　完了")

def train_data(folder, batch_size, num_epoch):
    window.write_event_value("-UPDATE-", "学習　開始\n")
    train(folder, batch_size, num_epoch)
    window.write_event_value("-PRINT-", "学習　完了\n")

def predict_data(folder, model):
    window.write_event_value("-UPDATE-", "画像を分割する\n")
    devide_images(folder)
    window.write_event_value("-PRINT-", "分割した画像に対して推論　開始\n")
    predict(folder, model)
    window.write_event_value("-PRINT-", "推論　完了\n")
    window.write_event_value("-PRINT-", "分割した画像を元に戻す\n")
    combine_images(folder)
    window.write_event_value("-PRINT-", "完了\n")

    
def devide_images(folder):
    project_folder = os.getcwd() + os.sep + folder
    in_subfolders = ["3_test_images_origin", "3_test_labels_origin"]
    out_subfolders = ["4_test_images", "4_test_labels", "5_predict_images"]

    for out_folder in out_subfolders:
        files = glob.glob(project_folder + os.sep + out_folder + os.sep + "*")
        window.write_event_value("-PRINT-", f"{out_folder}内のファイルを全削除")
        for file in files:
            os.remove(file)
    window.write_event_value("-PRINT-", "")

    for in_folder, out_folder in zip(in_subfolders, out_subfolders):
        files = glob.glob(folder + os.sep + in_folder + os.sep + "*")
        for file in files:
            filename = os.path.basename(file)                               # フルパスからファイル名を取得
            img_origin = cv2.imread(file)                                   # 画像読み込み
            height, width = img_origin.shape[:2]                            # 高さと幅
            xcnt ,ycnt = math.ceil(width/IMAGE_SIZE), math.ceil(height/IMAGE_SIZE)      # 256で割った数
            image = np.zeros((ycnt*IMAGE_SIZE, xcnt*IMAGE_SIZE,3), np.uint8)      # 256で割り切れるサイズのベース画像
            image[:height, :width] = img_origin                             # その左上に画像を貼る
            
            for i in range(xcnt):
                x1 = i * IMAGE_SIZE
                x2 =(i+1)*IMAGE_SIZE
                for j in range(ycnt):
                    y1 = j * IMAGE_SIZE
                    y2 = (j+1)*IMAGE_SIZE

                    roi = image[y1:y2, x1:x2]
                    roi_filename = f"{i},{j},{filename}"
                    cv2.imwrite(folder + os.sep + out_folder + os.sep + roi_filename, roi)

def combine_images(folder):
    # project_folder = os.getcwd() + os.sep + folder
    project_folder = folder
    original_subfolder = "3_test_images_origin"
    in_subfolder = "5_predict_images"
    out_subfolder = "6_predict_images_combine"

    original_files = glob.glob(project_folder + original_subfolder + os.sep + "*")

    #out_files = glob.glob(project_folder + out_subfolder + os.sep + "*")
    #for file in out_files:
    #    os.remove(file)

    predict_files = glob.glob(project_folder + in_subfolder + os.sep + "*")
    for ori_file in original_files:
        image = cv2.imread(ori_file)
        height, width = image.shape[:2]                            # 高さと幅
        xcnt ,ycnt = math.ceil(width/IMAGE_SIZE), math.ceil(height/IMAGE_SIZE)      # 256で割った数        
        image = np.zeros((ycnt*IMAGE_SIZE, xcnt*IMAGE_SIZE,3), np.uint8)      # 256で割り切れるサイズのベース画像

        filename = os.path.basename(ori_file)                               # フルパスからファイル名を取得
        pred_filename = "pred_" + filename
        for pred_file in predict_files:
            if filename in pred_file:
                roi = cv2.imread(pred_file)
                elms = pred_file.split(",")
                i , j = int(elms[1]), int(elms[2])
                x1 = i*IMAGE_SIZE
                x2 = (i+1)*IMAGE_SIZE
                y1 = j*IMAGE_SIZE
                y2 = (j+1)*IMAGE_SIZE
                image[y1:y2, x1:x2] = roi
        image = image[:height, :width]                                      # 元のサイズにトリミング
        cv2.imwrite(project_folder + out_subfolder + os.sep + pred_filename, image)

def img2bin(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                    # グレースケールにする
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)            # 二値化（0と255）
    bw = bw//255                                                    # 0と1にする
    bw = bw.flatten()                                               # 一次元にする
    return bw

def compareImg(project_folder, img1, img2):
    window.write_event_value("-UPDATE-", "")
    img_label = cv2.imread(img1)
    img_pred = cv2.imread(img2)
    if img_label.shape == img_pred.shape:
        window.write_event_value("-PRINT-", os.path.split(img1)[1] + "と")
        window.write_event_value("-PRINT-", os.path.split(img2)[1] + "を比較します")
    else:
        window.write_event_value("-PRINT-", "二つの画像のサイズが違うため")
        window.write_event_value("-PRINT-", "比較できません")
        return
        
    label_1bit = img2bin(img_label)                                 # 0と1の二値化


    
    pred_1bit = 2*img2bin(img_pred)                                 # 0と2の二値化
    compare_2bit = label_1bit + pred_1bit                           # 足し算することで0と1と2と3になる
    # ラベルの意味
    # 0  正常を正常と判断した(黒)
    # 1　異常を正常と見逃した（赤）
    # 2　正常を異常と過検出した（緑）
    # 3  異常を正しく異常と検出した（白）
    
    # 0～3の値に従い色を置く（ファンシーインデックス）
    colors = np.array([(0,0,0), (0,0,255), (0,255,0), (255,255,255)], np.uint8)     # ラベルに応じた色
    result = colors[compare_2bit]                                                   # ファンシーインデックス
    result = result.reshape(img_pred.shape)                                         # 画像サイズにリシェイプ

    tn = sum(compare_2bit==0)
    fn = sum(compare_2bit==1)
    fp = sum(compare_2bit==2)
    tp = sum(compare_2bit==3)
    
    #cv2.imshow("result", result)
    cv2.imwrite(project_folder + os.sep + "result.png", result)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

    window.write_event_value("-PRINT-", "")
    window.write_event_value("-PRINT-", "混同行列")
    window.write_event_value("-PRINT-", f"              異常と判定    正常と判定")
    window.write_event_value("-PRINT-", f"本当は異常   {tp}        {fn}")
    window.write_event_value("-PRINT-", f"本当は正常   {fp}        {tn}")


gui = Gui()
window = gui.window
logger = window["-ML-"]

def main():
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == "-FOLDER_CREATE-":
            folder = values["-FOLDER0-"]
            window.start_thread(lambda: gui.folder_create(folder), end_key="-CREATE_END-")
        elif event == "-FOLDER_CHECK-":
            folder = values["-FOLDER-"]
            window.start_thread(lambda: gui.folder_check(folder), end_key="-CHECK_END-")
        elif event == "-FOLDER_OK-":
            window["-RESULT_FOLDER-"].update(values["-FOLDER_OK-"])
        elif event == "-IMG_START-":
            if gui.is_folder_ok:
                size = 255
                folder = values["-FOLDER-"]
                window.start_thread(lambda: data_augumentation(folder, size), end_key="-IMG_END-")
            else:
                logger.update("先にフォルダーチェックをしてください")
        elif event == "-TRAIN_START-":
            logger.update("")
            if gui.is_folder_ok:
                batch_size = int(values["-BATCH-"])
                num_epoch =  int(values["-EPOCH-"])
                folder = values["-FOLDER-"]
                window.start_thread(lambda: train_data(folder, batch_size, num_epoch), end_key="-TRAIN_END-")
            else:
                logger.print("先にフォルダーチェックをしてください")
        elif event == "-PREDICT_START-":
            logger.update("")
            if gui.is_folder_ok:
                model = values["-MODELS-"]
                if model != "":
                    window.start_thread(lambda: predict_data(folder, model), end_key="-PREDICT_END-")
            else:
                logger.print("先にフォルダーチェックをしてください")
        elif event == "-EVAL_START-":
            folder = values["-FOLDER-"]
            img1 = values["-LABEL-"]
            img2 = values["-PRED-"]
            if img1 != "" and img2 != "":
                window.start_thread(lambda: compareImg(folder, img1, img2), end_key="-COMPARE_END-")
        elif event == "-PRINT-":
            logger.print(values[event])
        elif event == "-UPDATE-":
            logger.update(values[event])

    window.close()

if __name__ == '__main__':
    main()