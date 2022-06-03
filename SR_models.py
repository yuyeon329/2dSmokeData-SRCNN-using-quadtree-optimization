import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('upsample', 'nearest', 'nearest, bilinear, or pixelShuffler')
tf.app.flags.DEFINE_string('model', 'enhancenet', 'for now, only enhancenet supported')
tf.app.flags.DEFINE_string('recon_type', 'residual', 'residual or direct')
tf.app.flags.DEFINE_boolean('use_bn', False, 'for res_block_bn')
#tf.app.flags.DEFINE_integer('rate', 3, '')

FLAGS = tf.app.flags.FLAGS


class model_builder:
    def __init__(self):
        return

    def preprocess(self, images): ##전처리 함수..
        pp_images = images / 255.0 ##입력으로 들어온 이미지를 255.0으로 나눠주고,,
        ## simple mean shift
        pp_images = pp_images * 2.0 - 1.0
        #pp_images = images
        return pp_images ##전처리 된 이미지를 리턴
    
    def postprocess(self, images):
        pp_images = ((images + 1.0) / 2.0) * 255.0   #여기는 마지막에 다시 돌려놓는거!
        #pp_images = images
        return pp_images
    
    def tf_nn_lrelu(self, inputs, a=0.2):
        with tf.name_scope('lrelu'):
            x = tf.identity(inputs)
            return (0.5 * (1.0 + a)) * x + (0.5 * (1.0 - a)) * tf.abs(x)
    
    def tf_nn_prelu(self, inputs, scope):
        # scope like 'prelu_1', 'prelu_2', ...
        with tf.variable_scope(scope):
            alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
            pos = tf.nn.relu(inputs)
            neg = alphas * (inputs - tf.abs(inputs)) * 0.5

            return pos + neg
    
    def res_block(self, features, out_ch, scope):
        input_features = features ##첫 컨볼루션이 끝난 피쳐맵을 input__features에 저장
        with tf.variable_scope(scope):
            features = slim.conv2d(input_features, out_ch, 3, activation_fn=tf.nn.relu, normalizer_fn=None)
            features = slim.conv2d(features, out_ch, 3, activation_fn=None, normalizer_fn=None)
            
            return input_features + features ##첫 컨볼루션이 끝난 피쳐맵+conv2d 2번 거친 피쳐맵
 
    def res_block_bn(self, features, out_ch, is_training, scope): # bn-relu-conv!!!
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        
        # input_features already gone through bn-relu
        input_features = features ##첫 컨볼루션이 끝난 피쳐맵을 input_features에 저장
        with tf.variable_scope(scope):
            features = slim.conv2d(input_features, out_ch, 3, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)
            features = slim.conv2d(features, out_ch, 3, activation_fn=None, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)

            return input_features + features  ##첫 컨복루션이 끝난 피쳐맵 + conv2d 2번 거친 피쳐맵 더해줌 ->정보손실 최소화 위해
    
    def phaseShift(self, features, scale, shape_1, shape_2):
        X = tf.reshape(features, shape_1)
        X = tf.transpose(X, [0, 1, 3, 2, 4])

        return tf.reshape(X, shape_2)
    
    def pixelShuffler(self, features, scale=2):
        size = tf.shape(features)
        batch_size = size[0]
        h = size[1]
        w = size[2]
        c = features.get_shape().as_list()[-1]#size[3]

        channel_target = c // (scale * scale)
        channel_factor = c // channel_target

        shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
        shape_2 = [batch_size, h * scale, w * scale, 1]

        input_split = tf.split(axis=3, num_or_size_splits=channel_target, value=features) #features, channel_target, axis=3)
        output = tf.concat([self.phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

        return output

    def upsample(self, features, rate=2):    #rate = 사이즈 배수 ##2배로 업셈플링
        if FLAGS.upsample == 'nearest':
            return tf.image.resize_nearest_neighbor(features, size=[rate * tf.shape(features)[1], rate * tf.shape(features)[2]])
        elif FLAGS.upsample == 'bilinear':
            return tf.image.resize_bilinear(features, size=[rate * tf.shape(features)[1], rate * tf.shape(features)[2]])
        else: #pixelShuffler
            return self.pixelShuffler(features, scale=2)

    ### 일반 image recon_image train ###
    def recon_image(self, inputs, outputs):
        '''
        LR to HR -> inputs: LR, outputs: HR
        HR to LR -> inputs: HR, outputs: LR
        '''
        resized_inputs = tf.image.resize_bicubic(inputs, size=[tf.shape(outputs)[1], tf.shape(outputs)[2]]) ##인접한 16개의 화소를 통해 정보없는 pixel을 보간
        if FLAGS.recon_type == 'residual':
            recon_outputs = resized_inputs + outputs ##결과 최적화를 위해 인풋과 아웃푼 더해줌 (크기같아야 가능하니깐 앞에서 upsampling해준 것)
        else:
            recon_outputs = outputs

        resized_inputs = self.postprocess(resized_inputs) ##보간해준 놈을 원래값으로 다시 돌려주고,
        resized_inputs = tf.cast(tf.clip_by_value(resized_inputs, 0, 255), tf.uint8) ##원래값으로 돌려놓은 보간해준 놈을 tf.clip__by_value를 통해 텐서 안의 값이 지정된 범위를 넘지않게 해서 로스가 줄어들지 않는 불상사 방지
        #tf.summary.image('4_bicubic image', resized_inputs)

        recon_outputs = self.postprocess(recon_outputs)    # -1~1 -> 0~255 변환

        return recon_outputs, resized_inputs ##보간한 low image랑 컨볼루션돌고 업셈플링해준 low image더한 놈이랑, 그냥 보간만 해주고 값 돌려놓은 놈 리턴


    ### our text image recon_image train ###
    # def recon_image(self, inputs, outputs):
    #     '''
    #     LR to HR -> inputs: LR, outputs: HR
    #     HR to LR -> inputs: HR, outputs: LR
    #     '''
    #     resized_inputs = tf.image.resize_bicubic(inputs, size=[tf.shape(outputs)[1],
    #                                                            tf.shape(outputs)[2]])  ##인접한 16개의 화소를 통해 정보없는 pixel을 보간
    #     if FLAGS.recon_type == 'residual':
    #         recon_outputs = resized_inputs + outputs  ##결과 최적화를 위해 인풋과 아웃푼 더해줌 (크기같아야 가능하니깐 앞에서 upsampling해준 것)
    #     else:
    #         recon_outputs = outputs
    #
    #     #resized_inputs = self.postprocess(resized_inputs)  ##보간해준 놈을 원래값으로 다시 돌려주고,
    #     resized_inputs = tf.cast(tf.clip_by_value(resized_inputs, -1, 1),
    #                              tf.float32)  ##원래값으로 돌려놓은 보간해준 놈을 tf.clip__by_value를 통해 텐서 안의 값이 지정된 범위를 넘지않게 해서 로스가 줄어들지 않는 불상사 방지
    #     # tf.summary.image('4_bicubic image', resized_inputs)
    #
    #     #recon_outputs = self.postprocess(recon_outputs)  # -1~1 -> 0~255 변환
    #
    #     return recon_outputs, resized_inputs  ##보간한 low image랑 컨볼루션돌고 업셈플링해준 low image더한 놈이랑, 그냥 보간만 해주고 값 돌려놓은 놈 리턴

    ### model part
    '''
    list:
    enhancenet
    '''
    def enhancenet(self, inputs, is_training): ##로우 이미지, 트레이닝 여부
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None):
            
            features = slim.conv2d(inputs, 64, 3, scope='conv1')   #1 ##TF-Slim을 이용해 간단히 구현한 레이어 //
            #FLAGS는 우리가 호출할때 따로 설정 안해주면 기본값이 쓰여 위에 기본값이 있구! use_bn = false
            for idx in range(10):
                if FLAGS.use_bn:
                    features = self.res_block_bn(features, out_ch=64, is_training=is_training, scope='res_block_bn_%d' % (idx)) ##첫 컨복루션이 끝난 피쳐맵 + conv2d 2번 거친 피쳐맵
                else:
                    features = self.res_block(features, out_ch=64, scope='res_block_%d' % (idx)) ##첫 컨볼루션이 끝난 피쳐맵+conv2d 2번 거친 피쳐맵
            ##처음엔 첫 컨볼루션 거친 결과값이 들어가지만, 그 다움부턴 features가 리턴값이기 때문에 이전값이 계속 돎
            ##그렇게 각각 2번씩 * 10번 반복해서 스무번을 돕니다.

            ##그렇게 컨볼루션층을 20번 돌아버린 피쳐맵을...
            ##컨볼루션 층을 거치면.. 크기가 작아지니깐.. 다시 크기를 키워야하므로.. upsampling...
            features = self.upsample(features) ##업셈플 된 놈을.. 4번 더 컨볼루션층을 거치고.. 이넘을 리턴..!
            features = slim.conv2d(features, 64, 3, scope='conv2')
            
            # features = self.upsample(features)   #this is x4
            features = slim.conv2d(features, 64, 3, scope='conv3')
            features = slim.conv2d(features, 64, 3, scope='conv4')
            outputs = slim.conv2d(features, 3, 3, activation_fn=None, scope='conv5')

        return outputs
    
    ########## Let's enhance our method!
    ### 일반 image train generator ###
    def generator(self, inputs, is_training, model='enhancenet'):
        '''
        LR to HR
        '''

        inputs = self.preprocess(inputs)     #0~255 -> -1~1 우선 저화질 이미지를 -1~1로 정규화

        with tf.variable_scope('generator'):
            if model == 'enhancenet':
                outputs = self.enhancenet(inputs, is_training) #저화질 이미지와, 트레이닝부울변수 넘겨줌 // outputs는 컨볼루션 층 돌고, 업셈플링까지 해준 low이미지
                # print(outputs.shape)

        outputs, resized_inputs = self.recon_image(inputs, outputs)     # 원본,결과 (x,h(x))

        return outputs, resized_inputs          #이 리턴값이 outputs이 바로 그 generate_image로 이어져

    #### our text image generator ###
    # def generator(self, inputs, is_training, model='enhancenet'):
    #     '''
    #     LR to HR
    #     '''
    #
    #     #inputs = self.preprocess(inputs)  # 0~255 -> -1~1 우선 저화질 이미지를 -1~1로 정규화
    #
    #     with tf.variable_scope('generator'):
    #         if model == 'enhancenet':
    #             outputs = self.enhancenet(inputs,
    #                                       is_training)  # 저화질 이미지와, 트레이닝부울변수 넘겨줌 // outputs는 컨볼루션 층 돌고, 업셈플링까지 해준 low이미지
    #             # print(outputs.shape)
    #
    #     outputs, resized_inputs = self.recon_image(inputs, outputs)  # 원본,결과 (x,h(x))
    #
    #     return outputs, resized_inputs  # 이 리턴값이 outputs이 바로 그 generate_image로 이어져
    
### test part

if __name__ == '__main__':
    
    batch_size = 64
    h = 512
    w = 512
    c = 3  # rgb #xyz
   
    high_images = np.zeros([batch_size, h, w, c]) # gt
    low_images = np.zeros([batch_size, int(h/2), int(w/2), c])


    input_high_images = tf.placeholder(tf.float32, shape=[batch_size, h, w, c], name='input_high_images')
    input_low_images = tf.placeholder(tf.float32, shape=[batch_size, int(h/2), int(w/2), c], name='input_low_images')

    model_builder = model_builder()
        
    outputs = model_builder.generator(input_low_images) ##model_builder 객체의 generator메소드에 input_low_image입력

    print(outputs)

    

