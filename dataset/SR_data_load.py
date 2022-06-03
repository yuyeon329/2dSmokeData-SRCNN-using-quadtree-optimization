import os
import time
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
try:
    import data_util
except ImportError:
    from dataset import data_util

tf.app.flags.DEFINE_float('SR_scale', 4.0, '')
tf.app.flags.DEFINE_boolean('random_resize', False, 'True or False')
tf.app.flags.DEFINE_string('load_mode', 'real', 'real or text') # real -> COCO DB
FLAGS = tf.app.flags.FLAGS

'''
image_path = '/where/your/images/*.jpg'
'''

def load_image(im_fn, hr_size):
    high_image = cv2.imread(im_fn, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)[:,:,::-1] # rgb converted

    resize_scale = 1 / FLAGS.SR_scale   ##1/4

    '''
    if FLAGS.random_resize:
        resize_table = [0.5, 1.0, 1.5, 2.0]
        selected_scale = np.random.choice(resize_table, 1)[0]
        shrinked_hr_size = int(hr_size / selected_scale)

        h, w, _ = high_image.shape
        if h <= shrinked_hr_size or w <= shrinked_hr_size:
            high_image = cv2.resize(high_image, (hr_size, hr_size))
        else:
            h_edge = h - shrinked_hr_size
            w_edge = w - shrinked_hr_size
            h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
            w_start = np.random.randint(low=0, high=w_edge, size=1)[0]
            high_image_crop = high_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :]
            high_image = cv2.resize(high_image_crop, (hr_size, hr_size))
    '''
    h, w, _ = high_image.shape
    if h <= hr_size or w <= hr_size: ##high__image의 크기가 h__r보다 작으면..
        high_image = cv2.resize(high_image, (hr_size, hr_size),interpolation=cv2.INTER_AREA)  ##h__R로 사이즈 재조정

    else: ##high__image크기가 더 클 경우
        h_edge = h - hr_size
        w_edge = w - hr_size
        h_start = np.random.randint(low=0, high=h_edge, size=1)[0] ##0~h_edge사이 난수 1개
        w_start = np.random.randint(low=0, high=w_edge, size=1)[0] ##0~w_edge사이 난수 1개
        high_image = high_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :] ##???

    low_image = cv2.resize(high_image, (0, 0), fx=resize_scale, fy=resize_scale) ##log__image는 0.n배로 사이즈 조정
                           #interpolation=cv2.INTER_AREA)
    '''
    high_image = cv2.imread(im_fn, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)[:, :, ::-1]  # rgb converted

    resize_scale = 1 / FLAGS.SR_scale

    '''
    if FLAGS.random_resize:
        resize_table = [0.5, 1.0, 1.5, 2.0]
        selected_scale = np.random.choice(resize_table, 1)[0]
        shrinked_hr_size = int(hr_size / selected_scale)

        h, w, _ = high_image.shape
        if h <= shrinked_hr_size or w <= shrinked_hr_size:
            high_image = cv2.resize(high_image, (hr_size, hr_size))
        else:
            h_edge = h - shrinked_hr_size
            w_edge = w - shrinked_hr_size
            h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
            w_start = np.random.randint(low=0, high=w_edge, size=1)[0]
            high_image_crop = high_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :]
            high_image = cv2.resize(high_image_crop, (hr_size, hr_size))
    '''
    h, w, _ = high_image.shape
    # if h <= hr_size or w <= hr_size:
    #     high_image = cv2.resize(high_image, (hr_size, hr_size),
    #                             interpolation=cv2.INTER_AREA)
    # else:
    #     h_edge = h - hr_size
    #     w_edge = w - hr_size
    #     h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
    #     w_start = np.random.randint(low=0, high=w_edge, size=1)[0]
    #     high_image = high_image[h_start:h_start + hr_size, w_start:w_start + hr_size, :]
    high_image = cv2.resize(high_image, dsize=(hr_size, hr_size), interpolation=cv2.INTER_AREA)
    low_image = cv2.resize(high_image, (0, 0), fx=resize_scale, fy=resize_scale)
    '''
    return high_image, low_image

def load_txtimage(im_fn, hr_size, batch_size):
    # filein = open(im_fn, 'r')
    # fileInfo = filein.readline()
    # fileInfo = fileInfo.split()
    # width, height = int(fileInfo[0]), int(fileInfo[1])
    #
    # image = np.zeros([height, width, 3])
    # while True:
    #     line = filein.readline()
    #     if not line:
    #         break
    #     line = line.split()
    #     x = int(line[0])  # col
    #     y = int(line[1])  # row
    #     if x != width and y != height:
    #         for i in range(3):
    #             image[y, x, i] = (float(line[2 + i]))
    # filein.close()
    # #high_image = cv2.resize(image, dsize=(hr_size, hr_size), interpolation=cv2.INTER_AREA)
    # high_image = cv2.resize(image, (0, 0), fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
    # fake = high_image
    #
    # return high_image, fake

    high_image = cv2.imread(im_fn, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)[:,:,::-1]
    resize_scale = 1 / FLAGS.SR_scale

    lr_size = int(hr_size * resize_scale) ##줄인 사이즈

    h, w, _ = high_image.shape

    hr_batch = np.zeros([batch_size, hr_size, hr_size, 3], dtype='float32')
    lr_batch = np.zeros([batch_size, lr_size, lr_size, 3], dtype='float32')

    h_edge = h - hr_size
    w_edge = w - hr_size

    passed_idx = 0
    max_iter = 200
    iter_idx = 0
    while passed_idx < batch_size:
        h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
        w_start = np.random.randint(low=0, high=w_edge, size=1)[0]

        crop_hr_image = high_image[h_start:h_start + hr_size, w_start:w_start+hr_size,:]

        if np.mean(crop_hr_image) < 250.0:
            hr_batch[passed_idx,:,:,:] = crop_hr_image.copy()
            crop_lr_image = cv2.resize(crop_hr_image, (0, 0), fx=0.25, fy=0.25,
                                       interpolation=cv2.INTER_AREA)
            lr_batch[passed_idx,:,:,:] = crop_lr_image.copy()
            passed_idx += 1
        else:
            iter_idx += 1

        if iter_idx == max_iter:
            crop_lr_image = cv2.resize(crop_hr_image, (0, 0), fx=0.25, fy=0.25,
                                       interpolation=cv2.INTER_AREA)
            while passed_idx < batch_size:
                hr_batch[passed_idx,:,:,:] = crop_hr_image.copy()
                lr_batch[passed_idx,:,:,:] = crop_lr_image.copy()
                passed_idx += 1
            return hr_batch, lr_batch

    return hr_batch, lr_batch

def get_record(image_path):
    images = glob.glob(image_path)
    print('%d files found' % (len(images))) ##파일 갯수 출력

    if len(images) == 0: ##만일 파일이 없으면 데이터셋 경로 확인
        raise FileNotFoundError('check your training dataset path')
    
    index = list(range(len(images)))                            #data 개수의 범위를 배열처리하여 index 에 할당

    while True:
        random.shuffle(index)  ##인덱스 섞구
        
        for i in index:                                         #index shuffle 하지 않았으므로 순차적으로 호출
            im_fn = images[i]
            
            yield im_fn #high_image, low_image   ##제네레이터를 위한 yield / im__fn이 _next__메서드의 반환값으로 나옴
            ##값을 함수 바깥으로 전달하면서 코드 실행을 함수 바깥에 양보


def generator(image_path, hr_size=512, batch_size=32): ##발생자/ 함수안에서 yield를 사용하면 함수는 제네레이터가 됨
    high_images = []
    low_images = []
    
    for im_fn in get_record(image_path):                                #image_path에 존재하는 data들을 하나씩 경로 뽑아내는 함수. 즉 im_fn = 1개의 이미지 데이터의 경로
        try:
            # TODO: data augmentation
            '''
            used augmentation methods
            only linear augmenation methods will be used:
            random resize, ...
            not yet implemented

            '''
            if FLAGS.load_mode == 'real':                                   #data = image
                high_image, low_image = load_image(im_fn, hr_size)          #이미지 데이터 전처리함수 . 설정 인수값(hr_size)보다 크면 축소,low_image 생성하기
                
                high_images.append(high_image)                              #추가하기
                low_images.append(low_image)

            elif FLAGS.load_mode == 'text':                                 #data = txt
                high_images, low_images = load_txtimage(im_fn, hr_size, batch_size)

            if len(high_images) == batch_size:                              #할당한 batch_size = 16 라면, 반복문이 16번째마다 반환함. but return이 아니므로 그 이후 반복문이 계속 실행됨
                yield high_images, low_images

                high_images = []                                            #배열초기화
                low_images = []

        except FileNotFoundError as e:                                      #예외처리
            print(e)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

def get_generator(image_path, **kwargs): ##generator함수 돌려줌
    return generator(image_path, **kwargs)      # **kwargs는 인수 포인터라서 여러개 인수 가지고 호출중

## image_path = '/where/is/your/images/*.jpg'
def get_batch(image_path, num_workers, **kwargs):
    try:
        generator = get_generator(image_path, **kwargs)   #여기까지 보기 아래는 자료구조 사용방법이야 구조는 queue 파이선 라이브러리 사용이라 안 봐도 알아서 잘 만들어줘!
                                                            #batch_size크기의 high,low  data 배열 반환받으며 트레이닝 시작 밑에는 데이터 자료구조 처리부분
        print("isstart?")
        enqueuer = data_util.GeneratorEnqueuer(generator, use_multiprocessing=False)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.001)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

if __name__ == '__main__':
    image_path = '/Users/user/Downloads/train2017/train2017/*.jpg' ##이미지 경로
    num_workers = 4 ##n개의 코어 처리
    batch_size = 32 ##한번의 batch마다 주는 데이터의 갯수
    input_size = 32 ##
    data_generator = get_batch(image_path=image_path,  ##
                               num_workers=num_workers,
                               batch_size=batch_size,
                               hr_size=int(input_size*FLAGS.SR_scale))

    for _ in range(100):
        start_time = time.time()
        data = next(data_generator)
        high_images = np.asarray(data[0])
        low_images = np.asarray(data[1])
        print('%d done!!! %f' % (_, time.time() - start_time), high_images.shape, low_images.shape)
        for sub_idx, (high_image, low_image) in enumerate(zip(high_images, low_images)):
            hr_save_path = '/Users/user/Downloads/EnhanceNet-Tensorflow-master/dataset/test_hr/%03d_%02d_hr_image.jpg' % (_, sub_idx)
            lr_save_path = '/Users/user/Downloads/EnhanceNet-Tensorflow-master/dataset/test_lr/%03d_%02d_sr_image.jpg' % (_, sub_idx)
            cv2.imwrite(hr_save_path, high_image[:,:,::-1])
            cv2.imwrite(lr_save_path, low_image[:,:,::-1])
