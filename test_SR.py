import os

import numpy as np
import tensorflow as tf

import SR_models
import utils
import cv2
import time

#테스트에 넣을 텍스트파일
#pathT = "C:/Users/jhkim/PycharmProjects/smoke/Original/scene1/obj" #512 scene1 smoke
#pathT = "/Users/jhkimMultiGpus/Desktop/scene-1/Original/obj/"
# pathT = "/Users/jhkimMultiGpus/Desktop/scene-2/Original/obj/"
#pathT =  "C:/Users/jhkim/PycharmProjects/velocity/obj"
#pathT = 'C:/Users/jhkim/PycharmProjects/velocity/obj256/obj'
#pathT = 'C:/Users/jhkim/PycharmProjects/velocity/new/test256txt'
#pathT = "C:/Users/jhkim/PycharmProjects/velocity/new/256image"
#pathT = 'C:/Users/jhkim/PycharmProjects/velocity/results/vel_train_nonflow/before/256txt/'
#pathT = "obj256/obj/"
pathT = "C:/Users/jhkim/PycharmProjects/smoke/postprocess/scene2/MergeTxt/"

tf.app.flags.DEFINE_string('model_path', '/where/your/model/folder', '')
tf.app.flags.DEFINE_string('image_path', '/where/your/test_image/folder', '')
tf.app.flags.DEFINE_string('save_path', '/where/your/generated_image/folder', '')
tf.app.flags.DEFINE_string('run_gpu', '0', '')
tf.app.flags.DEFINE_integer('scale', 2, '')

FLAGS = tf.app.flags.FLAGS

imgNum = []
scale = FLAGS.scale


def load_model(model_path):
    '''
    model_path = '.../where/your/save/model/folder'
    '''
    input_low_images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input_low_images')

    model_builder = SR_models.model_builder()

    generated_high_images, resized_low_images = model_builder.generator(input_low_images, is_training=False, model='enhancenet')

    generated_high_images = tf.cast(tf.clip_by_value(generated_high_images, 0, 255), tf.float32)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    gen_vars = [var for var in all_vars if var.name.startswith('generator')]

    saver = tf.train.Saver(gen_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_path = utils.get_last_ckpt_path(model_path)
    saver.restore(sess, ckpt_path)

    return input_low_images, generated_high_images, sess


def real_num_r_tree(data,width,height,num,domain_size,imgNums):
    num_st = str(num)
    name = "smoke_"+num_st
    for i, j in enumerate(imgNums):
        if imgNums[i] == 0:
            continue
        else:
            imgNums[i] -= 1
            name = "smoke_"+str(i)
            break

    asd = "/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/enhance/outputs/SmokeTestData"
    if not os.path.isdir(asd + "/" + name + "/"):
        os.mkdir(asd + "/" + name + "/")
    n_asd = asd + "/" + name + "/"
    tmp_save_path = n_asd + num_st+".png"

    data = np.array(data, dtype="float32")
    data = data / 255.0
    cv2.imwrite(tmp_save_path, data)
    file = open(n_asd + num_st + '.txt', 'w')
    file.write(domain_size[num])
    for x in range(width):
        for y in range(height):
            write_rgb = "{0} {1}".format(x, y)
            bgr = " {0} {1} {2}\n".format(data[y][x][2], data[y][x][1], data[y][x][0])
            file.write(write_rgb+bgr)
    file.close()


def real_num_r(data, width, height, num):
    name = "smoke_%03d" %num
    name2 = "smoke_img_%03d" %num
    #테스트 결과 txt 저장경로
    #asd = "/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/enhance/outputs/SmokeTestData" ####################
    #asd = "C:/Users/jhkim/PycharmProjects/smoke/enhance/outputs/SmokeTestData" #512->1024 smoke scene1
    #asd = "C:/Users/jhkim/PycharmProjects/velocity/enhance/outputs/VelocityTestData"
    #asd = 'C:/Users/jhkim/PycharmProjects/velocity/enhance256/outputs/VelocityTestData' #vel 256 ->512
    #asd = 'C:/Users/jhkim/PycharmProjects/velocity/enhance256(2nd)/outputs/VelocityTestData' #256,512로 트레이닝한 모델 테스트용(과적합됏던넘)
    #asd = 'C:/Users/jhkim/PycharmProjects/velocity/enhance256(3rd)/outputs/VelocityTestData'
    #asd = 'C:/Users/jhkim/PycharmProjects/velocity/new/enhance256/outputs3/VelocityTestData'
    #asd = 'C:/Users/jhkim/PycharmProjects/velocity/results/img_train_flow/after/outputs/VelocityTestData/'
    #asd = 'C:/Users/jhkim/PycharmProjects/smoke/after_test/scene1/'
    asd = 'C:/Users/jhkim/PycharmProjects/smoke/enhanced/part1/scene2/'
    if not os.path.isdir(asd + "/" + name + "/"):
        os.mkdir(asd + "/" + name + "/")

    if not os.path.isdir(asd):
        os.mkdir(asd)
    n_asd = asd + "/" + name + "/"
    tmp_save_path = asd + "/" + name2 +".jpg"
    print(tmp_save_path)
    #img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    cv2.imwrite(tmp_save_path, data)

    data = np.array(data,dtype="float32")
    data = data / 255.0

    file = open(n_asd + "/" + name + '.txt', 'w')
    w_h = str(width) + " " + str(height) + "\n"
    file.write(w_h)
    for x in range(width):
        for y in range(height):
            write_rgb = "{0} {1}".format(x, y)
            bgr = " {0} {1} {2}\n".format(data[y][x][2], data[y][x][1], data[y][x][0])
            file.write(write_rgb+bgr)
    file.close()
    #data = ((data + 1.0) / 2.0) * 255.0
    #cv2.imwrite(tmp_save_path, data)


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank imag
    image = np.zeros((height, width, 3), np.float64)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def Test_Image_Processing_txt_Tree():
    contents = []
    coments = []

    num = 0
    for path, dirs, files in os.walk(pathT):
        for file in files:
            num += 1
    print("\ncount : " + str(num))

    for i in range(num):
        j = 0
        # pwd = "/Users/jhkimMultiGpus/Desktop/scene-1/Quad-Tree Image&Txt/"
        pwd = "/Users/jhkimMultiGpus/Desktop/txtImages/"
        if i < 10:
            pwd = pwd + "smoke_00" + str(i) + "/smoke_00" + str(i) + ".txt/"
        elif i < 100:
            pwd = pwd + "smoke_0" + str(i) + "/smoke_0" + str(i) + ".txt/"
        else:
            pwd = pwd + "smoke_" + str(i) + "/smoke_" + str(i) + ".txt/"

        for path, dirs, files in os.walk(pwd):
            for file in files:
                if os.path.splitext(file)[1].lower() == '.txt':
                    asd = pwd + '/' + file
                    filein = open(asd, 'r')
                    fileInfo = filein.readline()
                    fileInfo = fileInfo.split()
                    width, height = int(fileInfo[0]), int(fileInfo[1])
                    domain_size = str(width*scale) + " " + str(height*scale) + " " + str(fileInfo[2]) + " " + \
                                   str(int(fileInfo[3])*scale) + " "+str(int(fileInfo[4])*scale) + " " + fileInfo[5] + "\n"

                    image = np.zeros([height, width, 3])
                    while True:
                        line = filein.readline()
                        if not line:
                            break
                        line = line.split()
                        x = int(line[0])  # col
                        y = int(line[1])  # row
                        if x != width and y != height:
                            for i in range(3):
                                image[y, x, i] = (float(line[2 + i]))
                    j += 1

                    filein.close()

                    contents.append(image)
                    coments.append(domain_size)
        imgNum.append(j)

    return contents,coments,imgNum


def Test_Image_Processing_txt():
    contents = []

    pwd = pathT
    idx = 0
    for path, dirs, files in os.walk(pwd):
        print("file len: %d" % len(files))
        for file in files:
            if os.path.splitext(file)[1].lower() == '.txt':
                asd = pwd + '/' + file
                print(asd)
                filein = open(asd, 'r')
                fileInfo = filein.readline()
                fileInfo = fileInfo.split()
                width, height = int(fileInfo[0]), int(fileInfo[1])
                image = np.zeros([height, width, 3])
                while True:
                    line = filein.readline()
                    if not line:
                        break
                    line = line.split()
                    x = int(line[0])  # col
                    y = int(line[1])  # row
                    if x != width and y != height:
                        for i in range(3):
                            image[255-y, x, i] = (float(line[2 + i]) * 255.0)
                filein.close()
                #domains.append(domain_size)
                image = cv2.flip(image, 0)
                contents.append(image)
                idx += 1
    #i += 1
    return contents


def init_resize_image(im):
    h, w, _ = im.shape
    size = [h, w]
    max_arg = np.argmax(size)
    max_len = size[max_arg]
    min_arg = max_arg - 1
    min_len = size[min_arg]

    maximum_size = 512 #1024
    if max_len < maximum_size:
        maximum_size = max_len
        ratio = 1.0
        return im, ratio
    else:
        ratio = maximum_size / max_len
        max_len = max_len * ratio
        min_len = min_len * ratio
        size[max_arg] = int(max_len)
        size[min_arg] = int(min_len)

        im = cv2.resize(im, (size[1], size[0]))

        return im, ratio


if __name__ == '__main__':
    # set your gpus usage
    start = time.time()  # start time save
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.run_gpu

    # get pre-trained generator model
    input_image, generated_image, sess = load_model(FLAGS.model_path)
    #image_size = 256 #원래 512
    # get test_image_list
    # test_image_list = utils.get_image_paths(FLAGS.image_path)

    #test_image_list,domain_size,imgNums = Test_Image_Processing_txt_Tree()  # if Quad-Tree Image
    test_image_list = Test_Image_Processing_txt() # if Normal Image

    #make save_folder
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    # do test
    for test_idx, test_image in enumerate(test_image_list):
        # loaded_image = cv2.imread(test_image)
        # loaded_image = cv2.resize(loaded_image, None, fx=1.0 / 2, fy=1.0 / 2, interpolation=cv2.INTER_AREA)
        loaded_image = test_image
        processed_image, tmp_ratio = init_resize_image(loaded_image)

        feed_dict = {input_image : [processed_image[:,:,::-1]]}

        output_image = sess.run(generated_image, feed_dict=feed_dict)

        output_image = output_image[0,:,:,:]
        idx = test_idx

        # image_name = os.path.basename(idx+".png")

        # tmp_save_path = os.path.join(FLAGS.save_path, 'SR_' + image_name)

        #cv2.imwrite(tmp_save_path, output_image)
        h, w, _ = output_image.shape

        #real_num_r_tree(output_image, w, h, test_idx, domain_size, imgNums)  # if Quad-Tree Image
        real_num_r(output_image, w, h, test_idx) # if Normal Image

        # plt.imsave(tmp_save_path, output_image)

        print('%d / %d completed!!!' % (test_idx + 1, len(test_image_list)))

print("********SUCCESS********")
print("time :", time.time() - start)


