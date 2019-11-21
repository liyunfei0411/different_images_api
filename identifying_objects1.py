import time
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import requests
import logging
from logging.handlers import RotatingFileHandler
import os
from collections import OrderedDict
from quguang import unevenLightCompensate
from illuminationChange import illum
import sys
import cupy


sys.setrecursionlimit(100000)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 配置日志信息
# 设置日志的记录等级
logging.basicConfig(level=logging.INFO)
# 创建日志记录器，指明日志保存的路径、每个日志文件的最大大小、保存的日志文件个数上限
file_log_handler = RotatingFileHandler("logs/log", maxBytes=1024*1024*100, backupCount=10)
# 创建日志记录的格式                 日志等级    输入日志信息的文件名 行数    日志信息
formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)d %(message)s')
# 为刚创建的日志记录器设置日志记录格式
file_log_handler.setFormatter(formatter)
# 为全局的日志工具对象（flask app使用的）添加日记录器
logging.getLogger().addHandler(file_log_handler)


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/bg')
def show_bg():
    return render_template("bg.html")


@app.route('/target')
def show_target():
    return render_template("target.html")


class ObjectRecognition:

    def __init__(self, bg, target, area=None):
        '''
        :param bg: 背景图
        :param target: 目标图
        :param top: 指定识别区域顶部位置
        :param bottom: 指定识别区域底部位置
        :param left: 指定识别区域左边位置
        :param right: 指定识别区域右边位置
        '''
        self.bg = bg
        self.target = target
        self.origin_bg = bg
        self.origin_target = target
        #self.target_name = target_name
        self.width = self.origin_target.shape[0]
        self.height = self.origin_target.shape[1]
        self.area = area
        # self.is_wb = None
        self.is_light = 0.0
        if self.area is not None:
            self.left_x = area["x_min"]
            self.left_y = area["y_min"]
            self.right_x = area["x_max"]
            self.right_y = area["y_max"]

    def clip_area(self):
        # 将背景图和识别图剪切指定的识别区域
        if self.area is not None:
            self.bg = self.bg[self.left_y:self.right_y, self.left_x:self.right_x]
            self.target = self.target[self.left_y:self.right_y, self.left_x:self.right_x]
            # cv2.imwrite("bg.jpg", self.bg)
            # cv2.imwrite("target.jpg", self.target)

    def check_light(self):
        # 检查参考图和识别图的对比度和明度的差距
        sum_bg = cupy.sum(self.bg)
        sum_target = cupy.sum(self.target)
        result = abs(sum_target - sum_bg)/sum_bg
        result = round(result, 2)
        return result

    # def check_bg(self):
    #     # 检查参考图是否是黑白图
    #     b, g, r = cv2.split(self.bg)
    #     if (b == g).all() and (b == r).all():
    #         return True
    #     else:
    #         return False

    def get_contours(self):
        '''

        :param bg: 背景图
        :param target: 目标图
        :return: 多余物体的位置
        '''
        if self.is_light > 0.5:
            blur_size = (21, 21)
            print("背景和识别图片对比度和亮度偏大识别准确率下降")
            app.logger.info("背景和识别图片对比度和亮度偏大识别准确率下降")
            blur_size = (21, 21)
            blocksize = 16
            self.bg = illum(self.bg)
            self.target = illum(self.target)
            self.bg = unevenLightCompensate(self.bg, blocksize)
            self.target = unevenLightCompensate(self.target, blocksize)
        else:
            blur_size = (15, 15)
        bg = cv2.GaussianBlur(self.bg, blur_size, 0)
        target = cv2.GaussianBlur(self.target, blur_size, 0)
        grayA = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        score, diff = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype('uint8')
        # if self.is_wb:
        #     print("背景是黑白图识别准确率会下降！")
        #     app.logger.info("背景是黑白图识别准确率会下降！")
        #     thresh = cv2.Canny(diff, 128, 256)
        # else:
        thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # ret, thresh = ...

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        return cnts


    def different(self):
        '''

        :param bg: 背景图
        :param target: 目标图
        :return: 相似度
        '''
        # 获取背景图的轮廓
        thresh_bg = cv2.Canny(self.bg, 0, 256)
        thresh_bg, contoursA, hierarchy = cv2.findContours(thresh_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 获取目标图的轮廓
        thresh_target = cv2.Canny(self.target, 0, 256)
        thresh_target, contoursB, hierarchy = cv2.findContours(thresh_target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算相似度
        sumbg = cupy.sum(thresh_bg)
        sumtest = cupy.sum(thresh_target)
        diff_score = round(float(sumbg/sumtest), 3)
        return diff_score


    def draw_min_rect_circle(self, img, cnts):  # conts = contours

        '''
        :param img: 目标图
        :param cnts: 多余物体的位置坐标
        :return: 画出多余的物体的位置矩形框和多余物体的位置
        '''

        img = np.copy(img)
        cnt_list = list()
        if self.area is not None:
            cv2.rectangle(img, (self.left_x, self.left_y), (self.right_x, self.right_y), (0, 255, 0), 2)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.area is not None:
                if self.left_x > 0:
                    x = self.left_x + x
                if self.left_y > 0:
                    y = self.left_y + y
            if self.area:
                if w > 30 and h > 30 and w <= (self.right_x - self.left_x) and h <= (self.right_y - self.left_y):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cnt_dict = {"x_min": x, "x_max": x + w, "y_min": y, "y_max": y + h}
                    cnt_list.append(cnt_dict)
            else:
                if w > 30 and h > 30:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cnt_dict = {"x_min": x, "x_max": x+w, "y_min": y, "y_max": y+h}
                    cnt_list.append(cnt_dict)
            if cnt_list:
                cnt_list_del = cnt_list
                for cnt_dict in cnt_list:
                    for cnt_dict_check in cnt_list_del:
                        origin_target = cv2.cvtColor(self.origin_target, cv2.COLOR_BGR2GRAY)
                        origin_bg = cv2.cvtColor(self.origin_bg, cv2.COLOR_BGR2GRAY)
                        target = origin_target[cnt_dict["y_min"]:cnt_dict["y_max"], cnt_dict["x_min"]:cnt_dict["x_max"]]
                        bg = origin_bg[cnt_dict_check["y_min"]:cnt_dict_check["y_max"],
                             cnt_dict_check["x_min"]:cnt_dict_check["x_max"]]
                        target_sum = cupy.sum(target)
                        bg_sum = cupy.sum(bg)
                        diff = abs(target_sum - bg_sum)
                        result = diff / bg_sum
                        if result < 0.08:
                            if cnt_dict in cnt_list:
                                cnt_list.remove(cnt_dict)
            if cnt_list:
                for cnt_dict in cnt_list:
                    cv2.rectangle(img, (cnt_dict["x_min"], cnt_dict["y_min"]), (cnt_dict["x_max"], cnt_dict["y_max"]),
                                  (0, 0, 255), 2)
            else:
                cnt_list = []
        return img, cnt_list


    def main(self):
        self.clip_area()
        self.is_light = self.check_light()
        # self.is_wb = self.check_bg()
        # 判断两个图片的相似度
        diff_score = self.different()
        # 找到多余物体的位置并画出来
        if diff_score < 0.95 or diff_score > 1:
            cnts = self.get_contours()
            draw_img, cnt_list = self.draw_min_rect_circle(self.origin_target, cnts)
            #  cv2.imshow("result", draw_img)
            #file_dir = os.path.join(os.path.abspath('.'), "results")
            #file_path = os.path.join(file_dir, self.target_name)
            #cv2.imwrite(file_path, draw_img)
            # cv2.waitKey(3000)
            if cnt_list:
                return diff_score, cnt_list
            else:
                return 0.0, None 
        else:
            # 两个图片相同返回零
            return 0.0, None


@app.route("/check", methods=["POST"])
def check_image():
    tic = time.time()
    try:
        # 获得背景图片和识别图片
        bg_url = request.form.get("bg_url")
        print("背景图片：", bg_url)
        target_url = request.form.get("image_url")
        # target_name = target_url.split('/')[-1]
        #if not target_name.endswith('.jpg'):
         #   target_name = target_name + '.jpg'
        print("识别图片", target_url)
    except Exception as e:
        print(e)
        app.logger.error("e")
        return jsonify({"error": "bg or image_url Missing parameter"})
    try:
        #获得识别区域
        area = request.form.get("area")
        if area is not None:
            area = eval(area)
    except Exception as e:
        print(e)
    if not area:
        area = None
    try:
        # 背景图片转换成cv2格式
        res_bg = requests.get(bg_url)
        image_bg = res_bg.content
        buf_bg = np.asarray(bytearray(image_bg), dtype=np.uint8)
        bg = cv2.imdecode(buf_bg, cv2.IMREAD_COLOR)
    except Exception as e:
        print(e)
        app.logger.error("no bg file")
        return jsonify({"error": "no bg file"})
    try:
        # 识别图片转换成cv2格式
        res_target = requests.get(target_url)
        image_bg = res_target.content
        buf = np.asarray(bytearray(image_bg), dtype=np.uint8)
        target = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        print(e)
        app.logger.error("no image file")
        return jsonify({"error": "no image file"})
    read_image_time = time.time()
    print('读取图片耗时：{:.2f}'.format(read_image_time-tic))
    try:
        # 识别图片获得相似度和多余物体的位置
        obj_reco = ObjectRecognition(bg, target, area)
        diff_score, coordinate_list = obj_reco.main()
        print("相似度:",diff_score)
        if coordinate_list:
            diff_dict = OrderedDict([("diff_score", diff_score), ("count", len(coordinate_list)), ("data", coordinate_list),("is_clean", False)])
            print("坐标：", coordinate_list)
            
            toc = time.time()
            print('spnet {:.2f}'.format(toc-tic))
            return jsonify(diff_dict)
        if diff_score == 0.0:
            diff_dict = OrderedDict([("diff_score", diff_score), ("count", 0), ("data", None),("is_clean", True)])
            print("坐标：None")
            toc = time.time()
            print('spnet {:.2f}'.format(toc - tic))
            return jsonify(diff_dict)

    except Exception as e:
        print(e)
        return jsonify({"error": "image or bg not image"})


if __name__ == '__main__':

    app.run(debug=False, host="0.0.0.0", port=5011)
