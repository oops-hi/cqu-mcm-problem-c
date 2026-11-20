# 输入数据

将赛题原始附件 `2026_CQU_MCM_Problem_C_DATA.csv` 放在本目录，或通过 `python run.py --data <文件路径>` 指定位置。请勿提交原始订单文件。

程序使用 `starttime`、`bikeid`、`geohashed_start_loc` 和 `geohashed_end_loc` 四列。原始时刻保留赛题提供的当地时间。日期覆盖与缺测由程序写入 `results/data_audit.csv`。
