# WakingAnalysis
### 実行方法
取得対象となるフォルダを検索するため、実行時に引数の入力が必要です。

e.g.

ディレクトリが以下のような構成であるとします。
```
root
|
|____20201201_node
            |______NS01_01_202011101149
            |                       |_output0.ply
            |                       .
            |                       .
            |                       |_outputN.ply
            |
            |______NS01_02_202011101149
            |                       |_output0.ply
            |                       .
            |                       .
            |                       |_outputN.ply
            |
            |______NS01_03_202011101149
                                    |_output0.ply
                                    .
                                    .
                                    |_outputN.ply
            
```
この場合、コマンドライン引数として与えるのは

「20201201_node」までのパスです。


python walk_time_series.py /root/20201201_node