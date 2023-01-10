from PIL import Image
from PIL import ImageChops
from PIL import ImageStat
import traceback
import logging
import json
import cv2

def init_logger():
    LOG_FORMAT = "%(asctime)s[%(levelname)s]:%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

def draw_diff_images(in_diff_file, path_one, path_two, output_one, output_two):
    '''
    读取经 compare_images 过程产生的 diff 图片，从diff中识别有哪些“物品” 从而产生灰度块
    in_diff_file 为经过两张图 diff 后产生的差值图像文件
    base_diff_file 为基于这张图画 diff 区块，可以选前面进行
    '''
    logging.info ("[draw_diff_images]Load diff:" + in_diff_file + ", base img:" + path_one + ", change img:" + path_two)

    # 使用 IMREAD_COLOR 的原因是输出到
    img_ori_1 = cv2.imread(path_one, cv2.IMREAD_COLOR)  # 读取RGB原图用于加载，后续在原图上画出diff图
    img_ori_2 = cv2.imread(path_two, cv2.IMREAD_COLOR)  
    img_gs = cv2.imread(in_diff_file, cv2.IMREAD_GRAYSCALE) #加载灰度图 后面 IMREAD_GRAYSCALE 参数用来干这个

    # 设置阈值 
    # 参考 https://www.datasciencelearner.com/cv2-threshold-method-implementation-python/
    # cv2.threshold(input_image, threshold_value, max_value, thresholding_technique)
    thresh = cv2.threshold(img_gs, 60, 255, cv2.THRESH_BINARY)[1]   

    # 寻找物品
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print (len(contours))

    box_list = []
    for cobj in contours:
        area = cv2.contourArea(cobj)
        perimeter = cv2.arcLength(cobj,True)  #计算轮廓周长
        approx = cv2.approxPolyDP(cobj,0.02*perimeter,True)  #获取轮廓角点坐标
        x, y, w, h = cv2.boundingRect(approx)
        box_list.append((x,y,w,h))

        cv2.rectangle(img_ori_1,(x,y),(x+w,y+h),(0,0,255),2)  #绘制边界框，下同
        cv2.rectangle(img_ori_2,(x,y),(x+w,y+h),(0,0,255),2)
        # 函数说明：https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
        # 
        
    #print (box_list)
    cv2.imwrite(output_one, img_ori_1)
    cv2.imwrite(output_two, img_ori_2)
    logging.info ("Check output, 1:" + output_one + ", 2:" + output_two)

def compare_images(path_one, path_two, diff_save_location, output_one, output_two):
    """
    比较图片，如果有不同则生成展示不同的图片
    @参数一: path_one: 第一张图片的路径
    @参数二: path_two: 第二张图片的路径
    @参数三: diff_save_location: 不同图的保存路径
    """
    logging.info ("[Do diff]Load diff image from:" + path_one + ", " + path_two)
    image_one = Image.open(path_one)
    logging.info ("[Do diff]Load P1 " + image_one.mode + ", " + str(image_one.size))
    image_two = Image.open(path_two)
    logging.info ("[Do diff]Load P2 " + image_two.mode + ", " + str(image_two.size))

    if(not image_one.mode == image_two.mode):
        # 如果两张图位深不一样，尝试把 img2 转化为跟 img1 一致
        logging.warning ("[Do diff]images mode not equals, try change im2 to compale with im1")
        image_two = image_two.convert(image_one.mode)
    try: 
        diff = ImageChops.difference(image_one, image_two)
        if diff.getbbox() is None:
        # 图片间没有任何不同则直接退出
            logging.info ("[Do diff]We are the same!")
        else:
            # 计算图片实际差异程度 
            stat = ImageStat.Stat(diff)
            diff_ratio = sum(stat.mean) / (len(stat.mean) * 255)
            logging.info ("[Do diff]Found diff image(" + str(diff_ratio) + "), view diff in path:" + diff_save_location)
            diff.save(diff_save_location)
            #print (diff.getbbox())

            # 使用 opencv 识别差异块 
            draw_diff_images(diff_save_location, path_one, path_two, output_one, output_two)
    except ValueError as e:
        text = ("表示图片大小和box对应的宽度不一致，参考API说明：Pastes another image into this image."
            "The box argument is either a 2-tuple giving the upper left corner, a 4-tuple defining the left, upper, "
            "right, and lower pixel coordinate, or None (same as (0, 0)). If a 4-tuple is given, the size of the pasted "
            "image must match the size of the region.使用2纬的box避免上述问题")
        #print("【{0}】{1}".format(e,text))
        traceback.print_exc()

def do_main():
    init_logger()
    compare_images('data/sample1.png', 'data/sample2.png', 'data/diff.png', 'data/diff1.png', "data/diff2.png")

if __name__=="__main__":
    do_main()