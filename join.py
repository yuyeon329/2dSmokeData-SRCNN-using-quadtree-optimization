import cv2
import os
import numpy as np
import math
from PIL import Image
import time
org_ads = 'C:/Users/jhkim/PycharmProjects/smoke/org(0925)/scene2_data/obj'
a = 'C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/obj_'
b = '/obj_'
c = '.txt'

save_ads = 'C:/Users/jhkim/PycharmProjects/smoke/join/scene2'

def txtRead(asd):
    fileInput = open(asd, 'r')
    u = fileInput.readline() #정보 읽고
    line = u.split()
    image = create_blank(int(line[0]), int(line[1]), (0, 0, 0)) #조각 크기만큼 3차원 배열 만듦
    image = np.array(image)#np

    while True:
        line = fileInput.readline()#0 0 0.000000 0.000000 0.000000
        if not line:
            break
        line = line.split()
        x = int(line[0])  # column   width #각 x좌표 따오고
        y = int(line[1])  # row  height #y좌표 따와서
        image[255 - y][x] = (float(line[2]), float(line[3]), float(line[4])) #조각 배열 채워주고,
    return image #조각 정보랑 조각 배열 리턴


def txtWrite(new_txt_ads, quadline, org_data, i, file): #d여기서 쓰까먹은 텍본 생성
    f = open(new_txt_ads, 'w')
    quadline = quadline.split()
    w = int(quadline[0]) * 8  # 32x32 조각의 크기
    h = int(quadline[1]) * 8  # 32x32 조각의 크기
    x = int(quadline[3]) * 8
    y = int(quadline[4]) * 8
    txt = create_blank(w, h, (0, 0, 0))  # merge_size*mergesize*3짜리 검은색 생성
    txt = np.array(txt)
    new_file_info = "{0} {1} {2} {3} {4} {5}\n".format(w, h, quadline[2], x, y, quadline[5])
    f.write(new_file_info)
    x_p = x - w
    y_p = y - h
    if x_p < 0 or y_p < 0:
        x_p = 0
        y_p = 0
    txt = org_data[x_p:x, y_p:y, :] #####x_p:x, y_p:y
    if not os.path.isdir(save_ads + '/' + str('%03d' % i) + "_image"):  # 만약 폴더가 없으면 만들어주고
        os.mkdir(save_ads + '/' + str('%03d' % i) + "_image")
    cv2.imwrite(save_ads + '/' + str('%03d' % i) + "_image/" + file + ".jpg", txt*255.0)
    for x in range(w):
        for y in range(h):
            write_xyz = "{0} {1}".format(x, y)
            xyz = " {:.6f} {:.6f} {:.6f}\n".format(txt[x][y][0], txt[x][y][1], txt[x][y][2]) ###210 #####
            f.write(write_xyz + xyz)
    f.close()


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float32)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image


def join(a, b, c, org_ads):
    for i in range(499):
        pwds = a + str('%03d' %i) + b + str('%03d' %i) + c #쿼드 장면당 조각 txt 파일 접근
        for path, dirs, files in os.walk(pwds):
            for file in files: #txt 파일 접근
                pads = pwds + '/' + file #txt 파일 접근
                quadfile = open(pads, 'r') #쿼드 한조각 txt 읽기용 오픈
                quadline = quadfile.readline() #32x32 정보값
                quad_i = os.path.splitext(file) #텍본 번호 따기
                if not os.path.isdir(save_ads + '/' + str('%03d' %i)):
                    os.mkdir(save_ads + '/' + str('%03d' %i) + '/')
                new_txt_ads = save_ads + '/' + str('%03d' %i) + '/' + quad_i[0] + '.txt' #쓰까먹은 텍본 저장 주소
                org_txt_ads = org_ads + '/' + 'smoke_' + str('%03d' %i) + '.txt' #org256x256 텍본 하나 주소
                org_data = txtRead(org_txt_ads)
                txtWrite(new_txt_ads, quadline, org_data, i, file)
        print(i)

join(a, b, c, org_ads)

