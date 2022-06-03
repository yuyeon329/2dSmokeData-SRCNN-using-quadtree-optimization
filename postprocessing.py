import cv2
import os
import numpy as np
import time

pathT = "C:/Users/jhkim/PycharmProjects/smoke/org(0925)/scene2_data/obj/" #merge된 이미지 결과 갯수 알기 위한 경로
# pathT = "/Users/jhkimMultiGpus/Desktop/scene-2/Original/obj/"

#pwd = "/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/enhance/outputs/SmokeTestData"
pwd = "C:/Users/jhkim/PycharmProjects/smoke/join/scene2"

#save_path = "/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/enhance/outputs/SmokeTestData/MergeData"
save_path = "C:/Users/jhkim/PycharmProjects/smoke/postprocess/scene2/MergeData"
save_txt_path = "C:/Users/jhkim/PycharmProjects/smoke/postprocess/scene2/MergeTxt"


def save_txt(txt, i, save_txt_path):
    fle = open(save_txt_path + '/' + str('%03d' % i) + '.txt', 'w')
    print(str(i)+ "******************************")
    res = 256
    new_info ="{0} {1}\n".format(str(res), str(res))
    fle.write(new_info)
    for x in range(res):
        for y in range(res):
            write_xy = "{0} {1}".format(x, y)
            xyz = " {:.6f} {:.6f} {:.6f}\n".format(txt[y][x][0]/255.0, txt[y][x][1]/255.0, txt[y][x][2]/255.0)
            fle.write(write_xy + xyz)
    fle.close()



def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float32)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image

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
        image[x][y] = (float(line[2]), float(line[3]), float(line[4])) #조각 배열 채워주고,
    return u, image #조각 정보랑 조각 배열 리턴


start = time.time()  # start time save #시작
print("START")
num = 0
Merge_size = 256 #완성될 크기

if not os.path.isdir(save_path): #만약 폴더가 없으면 만들어주고
    os.mkdir(pwd + "/MergeData")

for path, dirs, files in os.walk(pathT): #완성되어야 할 갯수 미리 세어줌
    for file in files:
        num += 1
print("num" + str(num)) #num = 499(s1, s2)

for i in range(num): #0~498동안
    pwd2 = pwd + '/'+ str('%03d' %i) #합쳐야할 이미지 들어있는 i번째 파일 하나 경로
    txt = create_blank(Merge_size, Merge_size, (0, 0, 0)) #merge_size*mergesize*3짜리 검은색 생성
    txt = np.array(txt)

    for path, dirs, files in os.walk(pwd2):#i번째 파일안에
        for file in files:#합쳐야할 txt본 하나씩 접근
            if os.path.splitext(file)[1].lower() == '.txt':#텍본이면
                asd = pwd2 + "/" + file #텍본경로 생성
                index, image = txtRead(asd)#(조각)텍본의 정보, np배열
                index = index.split()#정보 잘라주고
                #name = os.path.splitext(file)
                image *= 255.0 #조각 텍본 rgb용으로 곱하기

                # 0:xsize 1:ysize 2:depth 3:xposition 4:yposition 5:state 6:xsize 7:ysize
                # 128 128 3 512 192 FD

                x = int(index[3]) - int(index[0])  # row
                y = int(index[4]) - int(index[1])  # col
                if x < 0 or y < 0:
                    x = 0
                    y = 0
                txt[x:x+int(index[1]), y:y+int(index[0]), :] = image #차례로 병합된 사이즈 txt array에 합쳐줌    ######y,x 바꿔줌
    #txt = cv2.flip(txt, 0)
    if i < 10: #각각 이미지로 저장
        cv2.imwrite(os.path.join(save_path, "smoke_00" + str(i) + '.jpg'), txt)
        save_txt(txt, i, save_txt_path)
    elif i < 100:
        cv2.imwrite(os.path.join(save_path, "smoke_0" + str(i) + '.jpg'), txt)
        save_txt(txt, i, save_txt_path)
    else:
        cv2.imwrite(os.path.join(save_path, "smoke_" + str(i) + '.jpg'), txt)
        save_txt(txt, i, save_txt_path)
    print("smoke_" + str(i) + '.jpg')#머가저장됐는지 출력
print("END")
print("time :", time.time() - start)

