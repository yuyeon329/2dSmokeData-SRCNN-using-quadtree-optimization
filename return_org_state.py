import cv2
import os
import numpy as np


def check_data_first(a, b, c, save_check, threshold): #원본 주소 특성 ㅐ매 따로 first로 만들어준 check_data
    for i in range(499):
        pwds = a + str('%03d' %i) + b + str('%03d' %i) + c #노가다로 넣어줘서
        for path, dirs, files in os.walk(pwds):
            for file in files:
                pads = pwds + '/' + file  #텍스트 파일 하나하나 접근
                fileInput = open(pads, 'r') #텍본 하나 열어주고
                line = fileInput.readline() #텍본에 있는 키값 정보 읽어주고
                ii = os.path.splitext(file) #저장용 텍본 만들기 위한 변수 ###ii.txt 파일
                if not os.path.isdir(save_check+'/'+str('%03d' %i)):
                    os.mkdir(save_check+'/'+str('%03d' %i)+'/')
                f = open(save_check+'/'+str('%03d' %i)+'/'+ ii[0] + '.txt', 'w') #i번째 장면의 ii번째 txt
                line = line.split()
                w = int(line[0])
                h = int(line[1])
                x = int(line[3])*8
                y = int(line[4])*8
                new_file_info = "{0} {1} {2} {3} {4} {5}\n".format(w, h, line[2], x, y, line[5])
                f.write(new_file_info)
                while True:
                    line = fileInput.readline()
                    if not line:
                        break
                    line = line.split()
                    x_p = int(line[0])
                    y_p = int(line[1])
                    coordinate = "{0} {1}".format(x_p, y_p)
                    datalists = [0 for i in range(3)]

                    if x_p != w and y_p != h:
                        for z in range(3):
                            if float(line[z+2])/255.0 > threshold:
                                datalists[z] = 1
                            else:
                                datalists[z] = 0
                        bi = " {0} {1} {2}\n".format(datalists[0], datalists[1], datalists[2])
                        f.write(coordinate+bi)
                f.close()
                fileInput.close() #텍스트 하나 이진화 완료


def check_data(pwd, save_check, threshold):
    for i in range(499):
        pwds = pwd + '/' + str('%03d' %i) + '/'
        for path, dirs, files in os.walk(pwds):
            for file in files:
                pads = pwds + '/' + file  #텍스트 파일 하나하나 접근
                fileInput = open(pads, 'r') #텍본 하나 열어주고
                line = fileInput.readline() #텍본에 있는 키값 정보 읽어주고
                ii = os.path.splitext(file) #저장용 텍본 만들기 위한 변수 ###ii.txt 파일
                if not os.path.isdir(save_check+'/'+str('%03d' %i)):
                    os.mkdir(save_check+'/'+str('%03d' %i)+'/')
                f = open(save_check+'/'+str('%03d' %i)+'/'+ ii[0] + '.txt', 'w') #i번째 장면의 ii번째 txt
                line = line.split()
                w = int(line[0])
                h = int(line[1])
                new_file_info = "{0} {1} {2} {3} {4} {5}\n".format(w, h, line[2], line[3], line[4], line[5])
                f.write(new_file_info)
                while True:
                    line = fileInput.readline()
                    if not line:
                        break
                    line = line.split()
                    x_p = int(line[0])
                    y_p = int(line[1])
                    coordinate = "{0} {1}".format(x_p, y_p)
                    datalists = [0 for i in range(3)]

                    if x_p != w and y_p != h:
                        for z in range(3):
                            if float(line[z+2]) > threshold:
                                datalists[z] = 1
                            else:
                                datalists[z] = 0
                        bi = " {0} {1} {2}\n".format(datalists[0], datalists[1], datalists[2])
                        f.write(coordinate+bi)
                f.close()
                fileInput.close() #텍스트 하나 이진화 완료


#resize()는 함수 안에서 텍스트 넘파이로 바꿔주고 -> api-> 넘파이 다시 텍스트로 바꿔줌 ->리사이즈 된 텍스트 저장할 ㅇㅇ

def resize_txt(pwd, address2):  #resize할 txt주소,저장할 주소,화질
    for i in range(499):
        pwds = pwd + '/' + str('%03d' %i) + '/'
        for path, dirs, files in os.walk(pwds):
            #print(path)
            for file in files:#n.txt
                iii = os.path.splitext(file) #파일 번호 따기
                pads = pwds + '/' + file  # 텍스트 파일 하나하나 접근
                fileInput = open(pads, 'r')  # 텍본 하나 열어주고
                key = fileInput.readline()  # 한줄 읽
                key2 = key.split()
                res = int(key2[0])
                img = np.zeros((res, res, 3), np.float32)
                #print("res : " + str(res)+" " + file)

                while True:
                    line = fileInput.readline()
                    if not line:
                        break
                    #print(line)
                    line = line.split()
                    x = int(line[0])
                    y = int(line[1])

                    if x != res and y != res :
                        for z in range(3):
                            img[y, x, z] = (float(line[2+z]))
                fileInput.close()
                resized_img = cv2.resize(img, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                write_txt(resized_img, address2, res*2, res*2, i, int(iii[0]), key)


def write_txt(resized_img, address1, width, height, count, counts, key): #줄인 배열,저장주소,넓이,높이,폴더 번호,파일 번호
    if not os.path.isdir(address1+'/'+str('%03d' %count)):
        os.mkdir(address1+'/'+str('%03d' %count)+'/')
    fle = open(address1+'/'+str('%03d' %count)+'/'+str(counts)+'.txt', 'w') ##re_txt에서 키값읽고 지나간걸로 가져와서 바꿔야함
    key = key.split()
    w = int(key[0])*2
    h = int(key[1])*2
    new_info = "{0} {1} {2} {3} {4} {5}\n".format(w, h, key[2], key[3],key[4],key[5])
    fle.write(new_info)
    for x in range(width):
        for y in range(height):
            write_xyz = "{0} {1}".format(x, y)
            xyz = " {:.6f} {:.6f} {:.6f}\n".format(resized_img[y][x][0], resized_img[y][x][1], resized_img[y][x][2]) ###210
            fle.write(write_xyz + xyz)
    fle.close()

a = 'C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/obj_'
b = '/obj_'
c = '.txt'
save_check_first = 'C:/Users/jhkim/PycharmProjects/smoke/return/scene2/first/binarized/obj/checked_obj'
save_resize_first = 'C:/Users/jhkim/PycharmProjects/smoke/return/scene2/first/binarized/obj/txt'

save_check_second = 'C:/Users/jhkim/PycharmProjects/smoke/return/scene2/second/binarized/obj/checked_obj'
save_resize_second = 'C:/Users/jhkim/PycharmProjects/smoke/return/scene2/second/binarized/obj/txt'

save_check_third = 'C:/Users/jhkim/PycharmProjects/smoke/return/scene2/third/binarized/obj/checked_obj'
save_resize_third = 'C:/Users/jhkim/PycharmProjects/smoke/return/scene2/third/binarized/obj/txt'

def main():
    check_data_first(a, b, c, save_check_first, 0.5) #임곗값미정 128 / 0.75 0.75
    resize_txt(save_check_first, save_resize_first)

    check_data(save_resize_first, save_check_second, 0.75)
    resize_txt(save_check_second, save_resize_second)

    check_data(save_resize_second, save_check_third, 0.75)
    resize_txt(save_check_third, save_resize_third)

main()
