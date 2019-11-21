import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import time
import os
from pymysql import *
from sys

base_path = os.path.dirname(sys.argv[0])
save_path = os.path.join(base_path, "result_image")
filename = os.path.join(save_path, "result_image.jpg")


def cap_image():
    while True:
        cap = cv2.VideoCapture("rtsp://192.168.10.253/live/0/MAIN")
        ret, image = cap.read()
        if ret:
            return image


def check_iswb(image):
    b, g, r = cv2.split(image)
    if (b == g).all() and (b == r).all():
        return True
    else:
        return False


def get_bg(is_wb):
    bg_dir = os.path.join(base_path, "bg_image")
    if is_wb:
        bg_path = os.path.join(bg_dir, "wb_bg.jpg")
        bg_image = cv2.imread(bg_path)
    else:
        bg_path = os.path.join(bg_dir, "bg.jpg")
        bg_image = cv2.imread(bg_path)
    return bg_image



class ObjectRecognition:

    def __init__(self, bg, target, area=None):

        self.bg = bg
        self.target = target
        self.origin_bg = bg
        self.origin_target = target
        # self.target_name = target_name
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
        if self.area is not None:
            self.bg = self.bg[self.left_y:self.right_y, self.left_x:self.right_x]
            self.target = self.target[self.left_y:self.right_y, self.left_x:self.right_x]

    def check_light(self):

        sum_bg = np.sum(self.bg)
        sum_target = np.sum(self.target)
        result = abs(sum_target - sum_bg) / sum_bg
        result = round(result, 2)
        return result

    def get_contours(self):

        blur_size = (15, 15)
        bg = cv2.GaussianBlur(self.bg, blur_size, 0)
        target = cv2.GaussianBlur(self.target, blur_size, 0)
        grayA = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        score, diff = ssim(grayA, grayB, full=True)
        print("相似度：", score)
        if score < 0.99:
            diff = (diff * 255).astype('uint8')
            thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # ret, thresh = ...

            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            return score, cnts
        else:
            return 0, None

    def draw_min_rect_circle(self, img, cnts):

        img = np.copy(img)
        cnt_list = list()
        if self.area is not None:
            cv2.rectangle(img, (self.left_x, self.left_y), (self.right_x, self.right_y), (0, 255, 0), 2)
        if cnts:
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                if self.area is not None:
                    if self.left_x > 0:
                        x = self.left_x + x
                    if self.left_y > 0:
                        y = self.left_y + y
                if w > 30 and h > 30:
                    cnt_dict = {"x_min": x, "x_max": x + w, "y_min": y, "y_max": y + h}
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
                            target_sum = np.sum(target)
                            bg_sum = np.sum(bg)
                            diff = abs(target_sum - bg_sum)
                            result = diff / bg_sum
                            print(result)
                            if result < 0.08:
                                if cnt_dict in cnt_list:
                                    cnt_list.remove(cnt_dict)
        if cnt_list:
            for cnt_dict in cnt_list:
                cv2.rectangle(img, (cnt_dict["x_min"], cnt_dict["y_min"]), (cnt_dict["x_max"], cnt_dict["y_max"]),
                              (0, 0, 255), 2)
        else:
            cnt_list = []
        cv2.imwrite(filename, img)
        print("保存检查环境整洁结果图片")
        return img, cnt_list

    def main(self):

        self.clip_area()
        diff_score, cnts = self.get_contours()
        draw_img, cnt_list = self.draw_min_rect_circle(self.origin_target, cnts)
        if cnt_list:
            return diff_score, cnt_list, draw_img
        else:
            return 0.0, None, draw_img

if __name__ == '__main__':
    conn = connect(host='192.168.10.229', port=3306, database='hall', user='root', password='root', charset='utf8')
    cur = conn.cursor()
    cap = cv2.VideoCapture("rtsp://admin:a1234567@192.168.10.253:554/h264/ch1/main/av_stream")
    i = 0
    test_time = time.time()
    while True:
        # time.sleep(5)
        start_time = time.time()
        if cap.isOpened():
            ret, image = cap.read()
        # image = cap_image()
            #cv2.imwrite("./logs/%d.jpg" % i, image)
        else:
            continue
        if i % 90 == 0:
            is_wb = check_iswb(image)
            bg = get_bg(is_wb)
            area = {"x_min": 1740, "x_max": 1908, "y_min": 630, "y_max": 900}
            obre = ObjectRecognition(bg, image, area)
            diff_score, cnt_list, img = obre.main()
            if diff_score:
                is_clean = 2
                print("环境不洁净")
            else:
                is_clean = 1
                print("环境整洁")
            params = [filename, is_clean]
            try:
                insert_sql = cur.execute("insert into hall_pic (path, is_clean) values(%s, %s);", params)
                conn.commit()
                if insert_sql:
                    print("数据库插入一条环境整洁数据")
            except Exception as e:
                print("环境整洁数据插入到数据库失败！")
                print(e)
                conn.rollback()
            # cv2.imwrite(os.path.join(save_path, str(i) + "result_image.jpg"), img)
            end_time = time.time()
            print("总耗时：{:.2f}".format(end_time - test_time))
        i += 1
    cur.close()
    conn.close()
