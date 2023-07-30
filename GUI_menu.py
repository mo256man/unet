import PySimpleGUI as sg
import numpy as np
import cv2
import glob
import os
from ml import *


class Gui():
    def __init__(self):
        self.is_folder_ok = False           # フォルダ内に必要なサブフォルダが揃っているかどうか

        layout_folder = [[sg.Text("プロジェクトフォルダを入力")],
                         [sg.InputText("sample", size=(40,1), key="-FOLDER-"),
                          sg.Button("check", key="-FOLDER_CHECK-"),
                          sg.Text("✕", key="-RESULT_FOLDER-")]]

        layout_img = [[sg.Text("size", size=(10,1)),
                       sg.InputText("256", size=(10,1), justification="right", key="-SIZE-"),
                       sg.Button("START", key="-IMG_START-")]]
        frame_img = [[sg.Frame("画像加工", layout_img)]]

        layout_teach = [[sg.Text("batch size", size=(10,1)),
                       sg.InputText("8", size=(10,1), justification="right", key="-BATCH-"),
                       sg.Button("START", key="-TRAIN_START-")],
                       [sg.Text("epoch", size=(10,1)),
                        sg.InputText("30", size=(10,1), justification="right", key="-EPOCH-")]]
        frame_teach = [[sg.Frame("学習", layout_teach)]]


        layout = [[layout_folder],
                  [frame_img],
                  [frame_teach],
                  [sg.Multiline("", size=(50,20),background_color="black", text_color="white", key="-ML-")]]
        self.window = sg.Window("UNet GUI", layout)

    def folder_check(self, folder):
        subfolders = ["1_image_origin", "1_label_origin",
                      "2_train_images", "2_train_labels",
                      "3_test_images", "3_test_labels", "3_predict_labels"]
        current_path = os.getcwd() + os.sep + folder
        window.write_event_value("-UPDATE-", "")

        # フォルダの存在確認
        if folder=="":
            window.write_event_value("-UPDATE-", f"フォルダ名を入力してください")
        elif not os.path.isdir(current_path):
            window.write_event_value("-UPDATE-", f"{folder}というフォルダは存在しません")
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
                else:
                    window.write_event_value("-FOLDER_OK-", "✕")

def data_augumentation(folder, size):
    project_folder = os.getcwd() + os.sep + folder
    in_subfolders = ["1_image_origin", "1_label_origin"]
    out_subfolders = ["2_images", "2_labels"]

    for out_folder in out_subfolders:
        files = glob.glob(project_folder + os.sep + out_folder + os.sep + "*")
        window.write_event_value("-PRINT-", f"{out_folder}内のファイルを全削除")
        for file in files:
            os.remove(file)
    window.write_event_value("-PRINT-", "")

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
        cnt = len(os.listdir(out_path))
        window.write_event_value("-PRINT-", f"　→ {cnt}個に拡張した\n")


gui = Gui()
window = gui.window
logger = window["-ML-"]

def main():
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == "-FOLDER_CHECK-":
            folder = values["-FOLDER-"]
            window.start_thread(lambda: gui.folder_check(folder), end_key="-CHECK_END-")
        elif event == "-FOLDER_OK-":
            window["-RESULT_FOLDER-"].update(values["-FOLDER_OK-"])
        elif event == "-IMG_START-":
            logger.update("")
            if gui.is_folder_ok:
                size = int(values["-SIZE-"])
                folder = values["-FOLDER-"]
                window.start_thread(lambda: data_augumentation(folder, size), end_key="-IMG_END-")
            else:
                logger.print("先にフォルダーチェックをしてください")
        elif event == "-TRAIN_START-":
            logger.update("")
            if gui.is_folder_ok:
                batch_size = int(values["-BATCH-"])
                num_epoch =  int(values["-EPOCH-"])
                folder = values["-FOLDER-"]
                logger.print("学習開始")
                window.start_thread(lambda: train(folder, batch_size, num_epoch), end_key="-IMG_END-")
                logger.print("完了")
            else:
                logger.print("先にフォルダーチェックをしてください")
        elif event == "-PRINT-":
            logger.print(values[event])
        elif event == "-UPDATE-":
            logger.update(values[event])

    window.close()

if __name__ == '__main__':
    main()