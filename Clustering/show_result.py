def show_result(result):
    """
     精度と実行時間を表示する

     Parameters
     ----------
     result : (float, float)
        精度と実行時間
    Returns
    -------
    None
    """
    (acc, time) = result
    print("accuracy is {}".format(acc))
    print("time is {} sec".format(time))