import cv2
import os
import numpy as np
import math
from PIL import Image
import time


#pwd = "/Users/jhkimMultiGpus/Desktop/scene-1/Original/obj/" #원본 txt
# pwd = "/Users/jhkimMultiGpus/Desktop/scene-2/Original/obj/"

#pwd = 'C:/Users/jhkim/PycharmProjects/smoke/s2/'
#pwd = 'C:/Users/jhkim/PycharmProjects/smoke/org(0925)/scene2_data/obj/'
pwd = 'C:/Users/jhkim/PycharmProjects/smoke/resize/part4/scene2/32/binarized/obj/txt/'

##pathT = "/Users/jhkimMultiGpus/Desktop/txtImages"
#trainpath = "/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/Train_data/"

pathT = 'C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/'
#pathT = 'C:/Users/jhkim/PycharmProjects/smoke/org256to64/scene2/'

image_size = 8  #8
image_original_size = 32 #32
Depth_num = int(math.log((image_original_size / image_size), 2))
end = (image_original_size//image_size)**2 #256/64 = 4, 4*4 = 16  #32/4 = 8 , 8*8 = 64

if image_size == 8: #bire
    end = 16
elif image_size == 64: #비교용 원본 256/64 = 4 , 4*4 = 16
    end = 16


def real_num(data,width,height, pwd2, pwd4, i):
    pwdq = pathT + "/" + pwd4 + "/" +pwd4 + "_" + str(i) + "result.txt/" + pwd4
    file = open(pwdq + '.txt', 'w')
    for y in range(height):
        for x in range(width):
            write_rgb = "{0} {1}".format(x,y)
            rgb=" {0} {1} {2}\n".format(data[x][y][0], data[x][y][1], data[x][y][2])
            file.write(write_rgb+rgb)
    file.close()

def search(dirname):
    images = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.jpg':
            images.append(filename)
    return images

def txtRead(pwd2, i):
    asd = pwd2 + str(i) + ".txt"
    fileInput = open(asd, 'r')
    list = fileInput.readline()
    list = list.split()
    return list

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float32)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image


# ImageToText(data, new_size, new_size, xi, pwd4, self.xLocation, self.yLocation, "FD", self.depth)
def ImageToText(data, width, height, num, name, xi, yi, D, Depth):
    data = np.array(data, dtype="float32")
    # Circle save
    file = open('C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/' + str(name) + "/" + str(name) + ".txt/" + str(num) + '.txt', 'w') # r=read, a=append, w=write
    w_h = str(width) + " " + str(height) + " " + str(Depth) + " " + str(xi) + " " + str(yi) + " " + D + "\n"
    file.write(w_h)
    for x in range(width):
        for y in range(height):
            write_rgb = "{0} {1}".format(x,y)
            bgr = " {0} {1} {2}\n".format(data[y][x][2], data[y][x][1], data[y][x][0]) ############################
            file.write(write_rgb+bgr)
    file.close()

"""
def ImageToText_circle(data,width,height, num, name, xi, yi, D, Depth):
    data = np.array(data,dtype="float32")
    fle = open('/Users/jhkimMultiGpus/Desktop/txtImages/' + 'cZ.txt', 'w')
    w_h = str(width) + " " + str(height) + " " + str(Depth) + " " + str(xi) + " " + str(yi) + " " + D + "\n"
    fle.write(w_h)
    for y in range(height):
        for x in range(width):
            write_rgb = "{0} {1}".format(y, x)
            bgr = " {0} {1} {2}\n".format(data[x][y][2], data[x][y][1], data[x][y][0])
            fle.write(write_rgb+bgr)
    fle.close()
"""
def image_cutting(file, image, m, width, height):
    x = 0
    y = 0
    i = 0

    name = os.path.splitext(file)
    if not os.path.isdir(pathT + "/" + name[0]):
        os.mkdir(pathT + "/" + name[0] + "/")
    if not os.path.isdir(pathT + "/" + name[0] + "/" + name[0] + ".txt"): ##폴더 안에 텍스트만 모아놓은 파일을 만들건데,, 그게 없으면
        os.mkdir(pathT + "/" + name[0] + "/" + name[0] + ".txt/")
    pathTT = pathT + "/" + name[0]

    while x < height:                # row   height
        while y < width:           # col   width
            test = image[x: x + m, y: y + m, 0:3]  # 16x16, 16x32, 16x48, ~ # (0,0)부터 시작해서 이미지 사이즈씩 증가하며 자름
            test *= 255.0 #rgb

            rgb = 0
            w = 0
            h = 0
            depth = Depth_num        # 512 = 0, 64 = 3 (위치 기준은 우측 아래)
            while w < m: #최소 노드 모두 돌며 rgb값 체크
                while h < m:
                    sum = 0
                    if int(test[w, h, 0]) > 0:
                        sum += 1
                    if int(test[w, h, 1]) > 0:
                        sum += 1
                    if int(test[w, h, 2]) > 0:
                        sum += 1
                    if sum > 0:
                        rgb += 1
                    h += 1
                h = 0
                w += 1

            # if (rgb / (image_size*image_size)) >= 0.02:
            if (rgb / (image_size*image_size)) > 0: #임계값과 비교하여 FD,ED 부여
                D = "FD"
                cv2.imwrite(os.path.join(pathTT, str(name[0]) + "_" + str(i) + '.jpg'), test)
                ImageToText(test, image_size, image_size, i, name[0], x+m, y+m, D, depth)

            else:
                D = "ED"
                cv2.imwrite(os.path.join(pathTT, str(name[0]) + "_" + str(i) + '.jpg'), test)
                ImageToText(test, image_size, image_size, i, name[0], x+m, y+m, D, depth)
            # Circles save
            y += m
            i += 1
        x += m
        y = 0


def txt_to_image_processing(): #txt한장 이미지로 만들어주고 /image__cuttiong() 만들어진 이미지 한장을 최소노드 크게 맞게 잘라주며, FD,ED,부여하여 이미지/txt모두 저장
    width = image_original_size
    height = image_original_size

    image = create_blank(width, height, (0, 0, 0)) ##rgb담을 w*h짜리 배열 생성
    image = np.array(image)

    for path, dirs, files in os.walk(pwd):
         for file in files:
             if os.path.splitext(file)[1].lower() == '.txt':
                 name = os.path.splitext(file)
                 asd = pwd + file
                 fileInput = open(asd, 'r')
                 line = fileInput.readline()
                 while True:
                     line = fileInput.readline()
                     if not line:
                         break
                     line = line.split()
                     x = int(line[0])        #col  width
                     y = int(line[1])        #row  height
                     if x != image_original_size and y != image_original_size:
                         image[(image_original_size-1)-y][x] = (float(line[2]), float(line[3]), float(line[4])) ###########################################
                 fileInput.close() #txt->image 만든 이미지 한장을
                 image_cutting(file, image, image_size, image_original_size, image_original_size) #(txt파일 한장,txt->img 한장, 자를사이즈, 원본 사이즈, 원본 사이즈)
                 print("processing - " + str(name[0]) + '.jpg')


def GetData(pwd, width, height):
    file_ = open(pwd, 'r')
    data = create_blank(width, height, (0, 0, 0))
    file_.readline()
    for w in range(width):
        for h in range(height):
            line = file_.readline()
            if not line:
                break
            line = line.split()
            x = int(line[0])  # column   width
            y = int(line[1])  # row  height
            if x != width and y != height:
                data[y][x] = (float(line[2]), float(line[3]), float(line[4]))
    file_.close()
    os.remove(pwd)
    return data


class QuadTree:
    def __init__(self):
        self.name = ""
        self.w = 0
        self.h = 0
        self.xLocation = 0
        self.yLocation = 0
        self.depth = 0
        self.density = ""

    def insert(self, w, h, x, y, depth, density):
        self.w = w
        self.h = h
        self.xLocation = int(x)
        self.yLocation = int(y)
        self.depth = int(depth)
        self.density = str(density)

    def merge(self, two, three, four, xi, yi, wi, hi, pwd2, pwd3, pwd4, i): #이미지 한장의 quad트리 merge
        filex = pwd3 + str(i) + "_" + str(xi)

        Nodelist = [xi, yi, wi, hi]
        Nodelist2 = [(self, xi), (two, yi), (three, wi), (four, hi)]
        if self.depth == two.depth and three.depth == four.depth and self.depth == three.depth:
                if self.density == "FD" and two.density == "FD" and three.density == "FD" and four.density == "FD":
                    if two.yLocation - self.yLocation == self.w and four.yLocation - three.yLocation == self.w and self.yLocation == three.yLocation: ###########다시이해
                        self.depth -= 1
                        org_size = self.w
                        new_size = self.w * 2
                        self.w = new_size
                        newimg = Image.new("RGB", (new_size, new_size))  # here
                        with Image.open(pwd3 + str(i) + "_" + str(xi) + ".jpg") as imgx:
                            newimg.paste(imgx, (0, 0, org_size, org_size))  # (0,0):(64,64)
                        with Image.open(pwd3 + str(i) + "_" + str(yi) + ".jpg") as imgy:
                            newimg.paste(imgy, (org_size, 0, new_size, org_size))
                        with Image.open(pwd3 + str(i) + "_" + str(wi) + ".jpg") as imgw:
                            newimg.paste(imgw, (0, org_size, org_size, new_size))
                        with Image.open(pwd3 + str(i) + "_" + str(hi) + ".jpg") as imgh:
                            newimg.paste(imgh, (org_size, org_size, new_size, new_size))
                        for k in Nodelist:
                            os.remove(pwd3 + str(i) + "_" + str(k)+".jpg")
                        newimg.save(filex + ".jpg")

                        data = create_blank(new_size, new_size, (0,0,0))
                        j = 0
                        for k in Nodelist:
                            asd = pwd2 + str(k)+".txt"
                            index = j % 2
                            index2 = 0 if j < 2 else org_size
                            data[index2:index2+org_size,index*org_size:index*org_size+org_size,:] = GetData(asd,org_size,org_size)
                            j += 1
                        x = self.xLocation - org_size + new_size
                        y = self.yLocation - org_size + new_size
                        self.xLocation += org_size
                        self.yLocation += org_size
                        ImageToText(data, new_size, new_size, xi, pwd4, x, y, "FD", self.depth)
                elif self.density == "ED" and two.density == "ED" and three.density == "ED" and four.density == "ED":
                    if two.yLocation - self.yLocation == self.w and four.yLocation - three.yLocation == self.w and self.yLocation == three.yLocation:
                        for k in Nodelist:
                            if os.path.isfile(pwd3 + str(i) + "_" + str(k)+".jpg"):
                                os.remove(pwd3 + str(i) + "_" + str(k)+".jpg")
                                os.remove(pwd2 + str(k) + ".txt")
                else:  ########################??mix?
                    for (k, k2) in Nodelist2:
                        if k.density == "ED":
                            if os.path.isfile(pwd3 + str(i) + "_" + str(k2) + ".jpg"):
                                os.remove(pwd3 + str(i) + "_" + str(k2) + ".jpg")
                                os.remove(pwd2 + str(k2) + ".txt")

def quad(pwd2, Nodenode, pwd3, pwd4, i, NodeIndex, pwd5): #NodeIndex:최소노드갯수
    Node = []  # Image
    for j in range(NodeIndex):
        line = txtRead(pwd2, j)
        # self, x, y, n, d, txt
        Qx = QuadTree()
        # x y Depth Den pwd
        Qx.insert(int(line[0]), int(line[1]), int(line[3]), int(line[4]), int(line[2]), line[5])
        # width height depth x,y
        Node.append(Qx)
    Nodenode.append(Node)

    if image_size == 64:
        nxt = 4
        xnn = 0
        ynn = 1
        wnn = 4
        hnn = 5
    elif image_size == 8: #4
        nxt = 4 #8
        xnn = 0
        ynn = 1
        wnn = 4 #8
        hnn = 5 #9
    num = 2 #쿼드트리니깐 2씩 증가하며 머지
    is_root = Depth_num
    while is_root != 0:
        one = xnn; two = ynn; three = wnn; four = hnn
        while four < end:
            Node[one].merge(Node[two], Node[three], Node[four], one, two, three, four, pwd2, pwd3, pwd4, i)
            if (one+num) % nxt != 0:
                one += num; two += num; three += num; four += num
            else:
                one += num+(num-1)*nxt; two += num+(num-1)*nxt; three += num+(num-1)*nxt; four += num+(num-1)*nxt
        xnn *= 2; ynn *= 2; wnn *= 2; hnn *= 2; num *= 2
        is_root -= 1

start = time.time()
print("START") ####시작
Node = []  # Images
txt_to_image_processing() #txt파일 하나 이미지 생성->이미지 한장 최소노드에 맞게 컷팅 ->  컷팅 된 최소 노드 임계값 비교하여 ED,FD --> 최소노드 킷값입력 -->나머지도 모두 처리
num = 0
for path, dirs, files in os.walk(pwd): #input에서
    for file in files:
        num += 1
for j in range(num):
    Node.append([]) #이미지 갯수 만큼 Node배열 생성
NodeIndex = end
for i in range(num):
 if i < 10:
     pwd2 = "C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/obj_00" + str(i) + "/obj_00" + str(i) + ".txt/"
     pwd3 = "C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/obj_00" + str(i) + "/obj_00"
     pwd4 = "obj_00" + str(i)
     pwd5 = "obj_00"
 elif i < 100:
     pwd2 = "C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/obj_0" + str(i) + "/obj_0" + str(i) + ".txt/"
     pwd3 = "C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/obj_0" + str(i) + "/obj_0"
     pwd4 = "obj_0" + str(i)
     pwd5 = "obj_0"
 else:
     pwd2 = "C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/obj_" + str(i) + "/obj_" + str(i) + ".txt/"
     pwd3 = "C:/Users/jhkim/PycharmProjects/smoke/quad/scene2/obj_" + str(i) + "/obj_"
     pwd4 = "obj_" + str(i)
     pwd5 = "obj_"
 quad(pwd2, Node, pwd3, pwd4, i, NodeIndex, pwd5)
 print("Quad-Tree : Smoke_" + str(i))
 i += 1

print("END")
print("time :", time.time() - start)
#원본 obj에서 각각 이미지로 만들어주면서 이미지 컷팅 해주고 최소노드로..
#그리고 그 최소노드 키값 입력 해주는거 끝나면 이미지 한장씩 돌면서 쿼드트리 연산 (자식노드의 성질에 따라 부모노드 병합하는 )

