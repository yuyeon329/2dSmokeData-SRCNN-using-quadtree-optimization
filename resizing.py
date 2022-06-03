#0911
#resize ex256->128
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import time


#0.0/0.1/0.01
scene1_org = 'C:/Users/jhkim/PycharmProjects/smoke/org(0925)/scene2_data/obj'
s1_256_checked_txt = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/256/binarized/obj/checked_obj'
s1_bi_128_txt = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/128/binarized/obj/txt'
s1_128_checked_txt = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/128/binarized/obj/checked_obj'
s1_bi_64_txt = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/64/binarized/obj/txt'
s1_64_checked_txt = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/64/binarized/obj/checked_obj'
s1_bi_32_txt = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/32/binarized/obj/txt'




def TxtToImage(address1, address2, res): #이미지화 할 텍스트주소 , 저장할 주소, 화질
    for path, dirs, files in os.walk(address1): #주소1
        count = -1
        for file in files:
            count += 1
            pads = address1 + '/' + file #주소1
            fileInput = open(pads, 'r')
            line = fileInput.readline()
            img = np.zeros((res,res,3),np.float32)

            while True:
                line = fileInput.readline()
                if not line:
                    break
                line = line.split()
                x = int(line[0])
                y = int(line[1])

                if x != res and y != res:
                    for z in range(3):
                        img[(res-1)-y, x, z] = float(line[z+2])*255.0
            fileInput.close()
            cv2.imwrite(address2+'/img_'+str('%03d' %count)+'.jpg',img) #주소2


#일단 새로운 틀 만들어놓고
#해당하는 원소 크기 비교 후
#0or1로 넣기
def check_data(address1, address2,res,threshold): #체크할 obj, 0,1로 된 텍스트 저장할 주소, 화질, 임계값
    for path, dirs, files in os.walk(address1): #
        count = -1
        for file in files:
            count += 1
            pads = address1 + '/' + file #
            fileInput = open(pads, 'r')
            line = fileInput.readline()
            f = open(address2+'/binarized_'+str('%03d' %count)+'.txt','w') ##
            resolution = "{0} {1}\n".format(res,res)
            f.write(resolution)
            while True:
                line = fileInput.readline()
                if not line:
                    break
                line = line.split()
                x = int(line[0])
                y = int(line[1])
                coordinate = "{0} {1}".format(x,y)
                datalists = [0 for i in range(3)]

                if x != res and y != res:
                    for z in range(3):
                        if float(line[z+2]) > threshold:
                            datalists[z] = 1
                        else:
                            datalists[z] = 0
                    bi = " {0} {1} {2}\n".format(datalists[0], datalists[1], datalists[2])#####################################################################
                    f.write(coordinate+bi)
            f.close()
            fileInput.close()


def bin_ndarray(ndarray, new_shape, operation):

    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray



#resize()는 함수 안에서 텍스트 넘파이로 바꿔주고 -> api-> 넘파이 다시 텍스트로 바꿔줌 ->리사이즈 된 텍스트 저장할 ㅇㅇ

def resize_txt(address1, address2, res):  #resize할 txt주소,저장할 주소,화질
    for path, dirs, files in os.walk(address1): #주소
        count = -1
        for file in files:
            count += 1
            pads = address1 + '/' + file #주소
            fileInput = open(pads,'r')
            line = fileInput.readline()
            img = np.zeros((res,res,3),np.float32)

            while True:
                line = fileInput.readline()
                if not line:
                    break
                line = line.split()
                x = int(line[0])
                y = int(line[1])

                if x != res and y != res :
                    for z in range(3):
                        img[y,x,z] = (float(line[2+z]))

            fileInput.close() #~ txt->numpy
            #api
            resized_img = bin_ndarray(img, new_shape=(res//2, res//2, 3), operation='avg')
            #~api

            write_txt(resized_img, address2, res//2, res//2, count)


def write_txt(resized_img, address1, width, height, count): #줄인 배열,저장주소,넓이,높이,count
    fle = open(address1 + '/obj_' + str('%03d' % count) + '.txt', 'w')
    resolution = "{0} {1}\n".format(width, height)
    fle.write(resolution)
    for x in range(width):
        for y in range(height):
            write_xyz = "{0} {1}".format(x, y)
            xyz = " {:.6f} {:.6f} {:.6f}\n".format(resized_img[y][x][0], resized_img[y][x][1], resized_img[y][x][2]) ###210
            fle.write(write_xyz + xyz)
    fle.close()


def Convert_image(address1, address2): #1 org 2 save (이미지 상하반전, bgr to rgb)
    for path, dirs, files in os.walk(address1): #주소1
        count = -1
        for file in files:
            count += 1
            pads = address1 + '/' + file  # 주소1
            img = cv2.imread(pads)
            flipped = cv2.flip(img, 0)
            img_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
            cv2.imwrite(address2 + '/img_' + str('%03d' % count) + '.jpg', img_rgb)  # 주소2


def main():
    start = time.time()
    print("START")  ####시작
    #256->128
    check_data(scene1_org, s1_256_checked_txt, 256, 0.05)
    resize_txt(s1_256_checked_txt, s1_bi_128_txt, 256)

    #128->64
    check_data(s1_bi_128_txt, s1_128_checked_txt, 128, 0.5)
    resize_txt(s1_128_checked_txt, s1_bi_64_txt, 128)

    #64->32
    check_data(s1_bi_64_txt, s1_64_checked_txt, 64, 0.5)
    resize_txt(s1_64_checked_txt, s1_bi_32_txt, 64)
    print("END")
    print("time :", time.time() - start)


#main() #bire

#visualize
"""
org128 = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/128/binarized/obj/txt'
img128 = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/128/binarized/img/bi_resized'

org64 = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/64/binarized/obj/txt'
img64 = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/64/binarized/img/bi_resized'

org32 = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/32/binarized/obj/txt'
img32 = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/32/binarized/img/bi_resized'

TxtToImage(org128, img128, 128)#128
TxtToImage(org64, img64, 64)#64
TxtToImage(org32, img32, 32)#32
"""

#only_resize

org = 'C:/Users/jhkim/PycharmProjects/smoke/org(0925)/scene2_data/obj'

re_128 = 'C:/Users/jhkim/PycharmProjects/smoke/only_resize/scene2/obj/128'
re_64 = 'C:/Users/jhkim/PycharmProjects/smoke/only_resize/scene2/obj/64'
re_32 = 'C:/Users/jhkim/PycharmProjects/smoke/only_resize/scene2/obj/32'
"""
resize_txt(org, re_128, 256)
resize_txt(re_128, re_64, 128)
resize_txt(re_64, re_32, 64)
"""

img_256 = 'C:/Users/jhkim/PycharmProjects/smoke/only_resize/scene2/img/256'
img_128 = 'C:/Users/jhkim/PycharmProjects/smoke/only_resize/scene2/img/128'
img_64 = 'C:/Users/jhkim/PycharmProjects/smoke/only_resize/scene2/img/64'
img_32 = 'C:/Users/jhkim/PycharmProjects/smoke/only_resize/scene2/img/32'
#visualize only_resize
TxtToImage(org, img_256, 256)
TxtToImage(re_128, img_128, 128)
TxtToImage(re_64, img_64, 64)
TxtToImage(re_32, img_32, 32)
