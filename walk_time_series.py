import glob
import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
# 平均と標準偏差の書き出し用
import json
import csv

file_path = ''
# json用
data = {}
# csv用
csv_data = []


def threeD_show_swing_for_csv(file_name, pcd_list):

    # 対象データから近似値の曲線に向けた垂直線の距離を計算する
    def get_intersection(x, y, re_polyfit):
        # 直線Lの式 ax + by + c = 0
        a = -re_polyfit[0]
        b = 1
        c = -re_polyfit[1]
        La = - a / b # 直線Lの傾き
        Lb = - c / b # 直線Lの切片

        # 点Aの座標(ax,ay)
        ax, ay = (x, y)

        # A点と直線Lの距離計算
        d = (abs(a*ax+b*ay+c))/((a**2+b**2)**(1/2)) # ヘッセの公式で距離計算
        
        hy = La*(La*(ay-Lb)+ax)/(La**2+1)+Lb
        if ay < hy:
            d = -d
        return d


    # 正規分布表示
    def create_normal_dist(title, amount):
        # 平均
        x_mean = np.mean(amount)
        # 標準偏差
        x_std = np.std(amount)
        
        # loc : 平均、scale : 標準偏差
        y = norm.pdf(amount, loc=x_mean, scale=x_std)
        print(" ----- amount : {}, y : {}".format(max(amount), max(y)))
        # 対象データのの表示設定
        plt.scatter(amount, y)

        # 平均値の表示設定
        plt.axvline(x_mean, color="g", linestyle='--', label="avg")

        # 標準偏差の表示設定
        calc_std = x_std*2
        print("std : {}".format(x_std))
        print("avg : {}".format(x_mean))
        # 平均と標準偏差の差を、平均から見た±それぞれの方向へ境界線を設定
        p_std = x_mean + x_std
        m_std = x_mean - x_std
        plt.axvline(p_std, color="r", linestyle='--', label="std")
        plt.axvline(m_std, color="r", linestyle='--')

        #print("{} | X : {} | Y : {}".format(file_name, amount, y))
        plt.legend()
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
        #plt.show()
        # json
        #return {"avg_intensity" : max(y), "std" : calc_std}
        # csv
        return ['avg', max(y), 'std', calc_std]

    x_record = pcd_list[:,0]
    y_record = pcd_list[:,1]
    z_record = pcd_list[:,2]

    ###### X軸にフィッティングさせた時のZ方向への揺れに関する処理 ######
    z = [i for i in range(len(z_record))]
    
    coe = np.polyfit(z, z_record, 1)
    y1 = np.poly1d(coe)(z)

    # ヘッセ
    diff_distance = []
    for i in range(len(z_record)):
        diff_distance.append(get_intersection(i, z_record[i], coe))
    
    plt.scatter(z, z_record)
    plt.plot(z, y1, color='green')
    #plt.show()
    # 正規分布の表示
    aggregate_x = create_normal_dist("Z", diff_distance)


    ###### Xデータに関する処理 ######
    """
    x = [i for i in range(len(x_record))]
    
    coe = np.polyfit(x, x_record, 1)
    y1 = np.poly1d(coe)(x)

    # ヘッセ
    diff_distance = []
    for i in range(len(x_record)):
        diff_distance.append(get_intersection(i, x_record[i], coe))
    
    plt.scatter(x, x_record)
    plt.plot(x, y1, color='green')
    plt.show()
    # 正規分布の表示
    create_normal_dist("X", diff_distance)
    """

    ###### Yデータに関する処理 ######
    y = [i for i in range(len(y_record))]
    
    coe = np.polyfit(y, y_record, 1)
    y1 = np.poly1d(coe)(y)

    # ヘッセ
    diff_distance = []
    for i in range(len(y_record)):
        diff_distance.append(get_intersection(i, y_record[i], coe))
    
    plt.scatter(y, y_record)
    plt.plot(y, y1, color='green')
    #plt.show()
    # 正規分布の表示
    aggregate_y = create_normal_dist("Y", diff_distance)
    # json出力を行うため、dictionaryを返す
    #return {'X' : aggregate_x, 'Y' : aggregate_y}
    # csv形式
    return ['X', aggregate_x, 'Y', aggregate_y]

def threeD_show_swing(file_name, pcd_list):

    def get_intersection(x, y, re_polyfit):
        # 直線Lの式 ax + by + c = 0
        a = -re_polyfit[0]
        b = 1
        c = -re_polyfit[1]
        La = - a / b # 直線Lの傾き
        Lb = - c / b # 直線Lの切片

        # 点Aの座標(ax,ay)
        ax, ay = (x, y)

        # A点と直線Lの距離計算
        d = (abs(a*ax+b*ay+c))/((a**2+b**2)**(1/2)) # ヘッセの公式で距離計算
        
        hy = La*(La*(ay-Lb)+ax)/(La**2+1)+Lb
        if ay < hy:
            d = -d
        return d


    # 正規分布表示
    def create_normal_dist(title, amount):
        # 平均
        x_mean = np.mean(amount)
        # 標準偏差
        x_std = np.std(amount)
        
        # loc : 平均、scale : 標準偏差
        y = norm.pdf(amount, loc=x_mean, scale=x_std)
        print(" ----- amount : {}, y : {}".format(max(amount), max(y)))
        # 対象データのの表示設定
        plt.scatter(amount, y)

        # 平均値の表示設定
        plt.axvline(x_mean, color="g", linestyle='--', label="avg")

        # 標準偏差の表示設定
        calc_std = x_std*2
        print("std : {}".format(x_std))
        print("avg : {}".format(x_mean))
        # 平均と標準偏差の差を、平均から見た±それぞれの方向へ境界線を設定
        p_std = x_mean + x_std
        m_std = x_mean - x_std
        plt.axvline(p_std, color="r", linestyle='--', label="std")
        plt.axvline(m_std, color="r", linestyle='--')

        #print("{} | X : {} | Y : {}".format(file_name, amount, y))
        plt.legend()
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
        plt.show()
        # json
        #return {"avg_intensity" : max(y), "std" : calc_std}
        # csv
        return ['avg', max(y), 'std', calc_std]

    x_record = pcd_list[:,0]
    y_record = pcd_list[:,1]
    z_record = pcd_list[:,2]

    ###### X軸にフィッティングさせた時のZ方向への揺れに関する処理 ######
    z = [i for i in range(len(z_record))]
    
    coe = np.polyfit(z, z_record, 1)
    y1 = np.poly1d(coe)(z)

    # ヘッセ
    diff_distance = []
    for i in range(len(z_record)):
        diff_distance.append(get_intersection(i, z_record[i], coe))
    
    plt.scatter(z, z_record)
    plt.plot(z, y1, color='green')
    plt.show()
    # 正規分布の表示
    aggregate_x = create_normal_dist("Z", diff_distance)


    ###### Xデータに関する処理 ######
    """
    x = [i for i in range(len(x_record))]
    
    coe = np.polyfit(x, x_record, 1)
    y1 = np.poly1d(coe)(x)

    # ヘッセ
    diff_distance = []
    for i in range(len(x_record)):
        diff_distance.append(get_intersection(i, x_record[i], coe))
    
    plt.scatter(x, x_record)
    plt.plot(x, y1, color='green')
    plt.show()
    # 正規分布の表示
    create_normal_dist("X", diff_distance)
    """
    ###### Yデータに関する処理 ######
    y = [i for i in range(len(y_record))]
    
    coe = np.polyfit(y, y_record, 1)
    y1 = np.poly1d(coe)(y)

    # ヘッセ
    diff_distance = []
    for i in range(len(y_record)):
        diff_distance.append(get_intersection(i, y_record[i], coe))
    
    plt.scatter(y, y_record)
    plt.plot(y, y1, color='green')
    plt.show()
    # 正規分布の表示
    aggregate_y = create_normal_dist("Y", diff_distance)
    # json出力を行うため、dictionaryを返す
    #return {'X' : aggregate_x, 'Y' : aggregate_y}
    # csv形式
    return [aggregate_x, aggregate_y]

### pcdファイルのデータを表示する ###
# file_name
#           表示対象ファイル名
# pcd_list
#           1ファイルから取得した[[X1, Y1, Z1], [X2, Y2, Z2], . . . [Xn, Yn, Zn]]の形状のデータ
def threeD_show_pcd_data(file_name, pcd_list):
    
    # X,Y,Zの各データに対する正規分布表示
    def create_normal_dist(title, amount):
        # 平均
        x_mean = np.mean(amount)
        # 標準偏差
        x_std = np.std(amount)

        # loc : 平均、scale : 標準偏差
        y = norm.pdf(amount, loc=x_mean, scale=x_std)
        
        # 対象データのの表示設定
        plt.scatter(amount, y)

        # 平均値の表示設定
        plt.axvline(x_mean, color="g", linestyle='--', label="avg")

        # 標準偏差の表示設定
        # 1. 平均と標準偏差の差
        diff_std = np.abs(x_mean - x_std)
        # 2. 平均と標準偏差の差を、平均から見た±それぞれの方向へ境界線を設定
        p_std = x_mean + diff_std
        m_std = x_mean - diff_std
        plt.axvline(p_std, color="r", linestyle='--', label="std")
        plt.axvline(m_std, color="r", linestyle='--')

        print("{} | X : {} | Y : {}".format(file_name, amount, y))
        plt.legend()
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
        plt.show()

    # XYZ座標上のデータの変位量表示
    amount = []
    current = []
    for record in pcd_list:
        if len(current) == 0:
            current = record
        else:
            diff = np.linalg.norm(record - current)
            amount.append(diff)
            current = record

    #print("amount >>> : {}".format(amount))
    x = [i for i in range(len(amount))]
    plt.scatter(x, amount)
    plt.plot(x, amount)
    plt.show()

    ### 変位量の正規分布 ###
    create_normal_dist(file_name, amount)


### pcdファイルのデータを表示する ###
# file_name
#           表示対象ファイル名
# pcd_list
#           1ファイルから取得した[[X1, Y1, Z1], [X2, Y2, Z2], . . . [Xn, Yn, Zn]]の形状のデータ
def show_pcd_data(file_name, pcd_list):

    # X,Y,Zの各データに対する正規分布表示
    def create_normal_dist(title, x_record):
        # 平均
        x_mean = np.mean(x_record)
        # 標準偏差
        x_std = np.std(x_record)

        # パターン1
        """
        x_array = []
        for i in range(len(x_record)):
            # loc : 平均、scale : 標準偏差
            #x_array.append(norm.pdf(x=x_record[i], loc=0, scale=1))
            x_array.append(norm.pdf(x=x_record[i], loc=x_mean, scale=x_std))
        """
        # パターン2
        # loc : 平均、scale : 標準偏差
        y = norm.pdf(x_record, loc=x_mean, scale=x_std)

        # 対象データのの表示設定
        plt.scatter(x_record, y)

        # 平均値の表示設定
        plt.axvline(x_mean, color="g", linestyle='--', label="avg")

        # 標準偏差の表示設定
        # 1. 平均と標準偏差の差
        diff_std = np.abs(x_mean - x_std)
        # 2. 平均と標準偏差の差を、平均から見た±それぞれの方向へ境界線を設定
        p_std = x_mean + diff_std
        m_std = x_mean - diff_std
        plt.axvline(p_std, color="r", linestyle='--', label="std")
        plt.axvline(m_std, color="r", linestyle='--')

        print("{} | X : {} | Y : {}".format(file_name, x_record, y))
        plt.legend()
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
        plt.show()

    diff = np.diff(pcd_list, axis=0)
    d = np.abs(diff)
    time = [i for i in range(len(d))]
    x_record = d[:,0]
    y_record = d[:,1]
    z_record = d[:,2]
    
    #print("x_record : {}, y_record : {}, z_record : {}".format(x_record, y_record, z_record))

    # グラフ定義
    # 散布図
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), facecolor = 'gray', subplot_kw = {'facecolor' : 'aqua'})
    axs[0].scatter(time, x_record, c='red', marker='.')
    axs[1].scatter(time, y_record, c='blue', marker='.')
    axs[2].scatter(time, z_record, c='green', marker='.')
    plt.show()
    
    # 3D表示
    fig = plt.figure()
    ax = Axes3D(fig)
    # X,Y,Z軸にラベルを設定
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot(x_record, y_record, z_record, marker=".", linestyle='None')
    plt.show()
    # 列ごとの分散
    dist_x = np.var(x_record)
    dist_y = np.var(y_record)
    dist_z = np.var(z_record)
    print(" << distributed value >> [x : {}], [y : {}], [z : {}]".format(dist_x, dist_y, dist_z))
    
    ### 正規分布 ###
    # xの正規分布表示
    title = file_name + ":" + "x"
    create_normal_dist(title, x_record)
    # yの正規分布表示
    title = file_name + ":" + "y"
    create_normal_dist(title, y_record)
    # zの正規分布表示
    title = file_name + ":" + "z"
    create_normal_dist(title, z_record)
    
# json組み立て
def json_el_add(type, array, init):
    if init:
        data[str(type)] = array
        return
    else:
        data[str(type)].append(array)

def csv_el_add(array):
    csv_data.append(array)

if __name__ == "__main__":
    args = sys.argv
    ### ルートにしたいディレクトリを引数に指定する ###
    f_path = str(args[1])
    type_str = ''
    aggre_result = []

    # jsonファイル名を作成
    if f_path.find('before') > 0:
        type_str = 'before'
    elif f_path.find('after') > 0:
        type_str = 'after'
    
    csv_file_path = f_path + "/" + type_str + ".csv"
    #file_path = f_path + "/" + type_str + ".json"
    print("output -- filename : {}".format(file_path))
    json_el_add(type_str, [], True)
    
    ### ファイル名若しくはフォルダ名を取得する ###
    # f_path 
    #           Trueの場合 : ルートとなるディレクトリパス
    #           Falseの場合 : 取得対象ファイルが格納されているフォルダまでのパス
    # own
    #           True : フォルダ名一覧を取得する
    #           False : ファイル名一覧を取得する
    def read_f(f_path, own):
        if own:
            dirs = os.listdir(f_path)
            folder = [f for f in dirs if os.path.isdir(os.path.join(f_path, f))]
            folder.sort()
            return folder
        else:
            files = os.listdir(f_path)
            files_file = [f for f in files if os.path.isfile(os.path.join(f_path, f))]
            files_file.sort()
            return files_file

    
    # 引数で受け取ったフォルダ配下のフォルダ名一覧を取得する
    target_dir = read_f(f_path, True)
    # フォルダ数分のloop
    for dir in target_dir:
        csv_h_dirname = dir[:4]
        print("csv_h_dirname : {}".format(csv_h_dirname))
        full_path = f_path + '/' + dir
        print(full_path)
        # [ルート + 各個フォルダ名]で指定したパス配下のファイル名一覧を取得する
        files = read_f(full_path, False)

        # ファイル数分のloop (= 人毎かつ骨格別のデータ表示)
        for file in files:
            print("----------file : {}".format(file))
            re_file = file[6:]
            print("----------refile : {}".format(re_file))
            # plyファイル以外を無視
            if file.find('.ply') < 0:
                continue
            # 指定する骨格以外も無視
            # 雑に緊急的に
            start = False
            if not start and re_file == '12.ply':
                start = True
            if not start and re_file == '18.ply':
                start = True
            if not start and re_file == '22.ply':
                start = True
            if not start and re_file == '26.ply':
                start = True
            if not start and re_file == '0.ply':
                start = True
            if not start and re_file == '5.ply':
                start = True
            if start:
                try:
                    # 各ファイルのpcdデータを取得する
                    pcd = o3d.io.read_point_cloud(full_path + '/' + file)
                    # 読み込んだままの状態では操作が出来ないため、[x, y, z]の形式にで配列に格納する
                    xyz_load = np.asarray(pcd.points)
                    #print("file >> pcd : {} >> {}".format(file, xyz_load))

                    ######## matplotlibでの表示 ########
                    # 1. X, Y, Zの各データの変位量に関する処理
                    #show_pcd_data(file, xyz_load)

                    # 2. XYZ座標上のデータ同士の変位量に関する処理
                    #threeD_show_pcd_data(file, xyz_load)

                    # 3. X,Y,Zそれぞれの方向への揺れを観測する
                    #aggre_result = threeD_show_swing(file, xyz_load)

                    # 4. X,Y,Zそれぞれの方向への揺れを観測する(画面表示なし)
                    aggre_result = threeD_show_swing_for_csv(file, xyz_load)
                    dir_n_file = dir + '_' + file

                    # json
                    #insert_dic = {dir_n_file : aggre_result}
                    #json_el_add(type_str, insert_dic, False)
                    
                    # csv
                    aggre_result.insert(0, [csv_h_dirname, file])
                    print("aggre_result >> : {}".format(aggre_result))
                    csv_array = []
                    for row in aggre_result:
                        for i in range(len(row)):
                            csv_array.append(row[i])
                    csv_el_add(csv_array)
                except Exception as ex:
                    print("Exception : {}".format(ex))
            
            #break # フォルダ内の全ファイルを取得したい場合はこのbreakをコメントアウトしてください
        
        
        print("csv : {}".format(csv_data))
        #np.savetxt(csv_file_path, csv_data, delimiter=',')
        with open(csv_file_path, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(csv_data)
        
        # json
        """
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)
        """
        #break # ルート内の全フォルダを取得したい場合はこのbreakをコメントアウトしてください
