# 更新日期：2019.04.06  --------- Max_acc: 0.9239
# 各用到的文件需先修改好
# test.prototxt 文件Accuracy层的名字为prob时会自动记录Max_acc


1.  ./train_net.sh                              # 训练(需准备好prototxt文件)
2.  python verify.py                            # 测试精度
