import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random


USE_CUSTOM_GEN = False

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--epoch_number", type = int, default = 5 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  )
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
w,h = args.input_width, args.input_height
#videoout = None
videoout = cv2.VideoWriter('output.avi',fourcc, 24.0, (w, h))

for idx, imgName in enumerate(images):
        in_img = cv2.imread(imgName)
        if USE_CUSTOM_GEN:
            output_height, output_width, channels = in_img.shape
	outName = imgName.replace( images_path ,  args.output_path )

        if USE_CUSTOM_GEN:
	    X = LoadBatches.getImageArr(imgName , output_width  , output_height  )
        else:
	    X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )

        if USE_CUSTOM_GEN:
            output_height = (output_height//16)*16 
            output_width = (output_width//16)*16 
            X = X[:,0:output_height, 0:output_width]

	pr = m.predict( np.array([X]) )[0]
        if USE_CUSTOM_GEN:
    	    pr = pr.reshape(( n_classes, output_height ,  output_width ) ).argmax( axis=0 )
	else:
            pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )

	seg_img = np.zeros( ( output_height , output_width , 3 ) )
	for c in range(n_classes):
		seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

        #import pdb; pdb.set_trace()
        if USE_CUSTOM_GEN:
            in_img = in_img[0:output_height, 0:output_width]
        else:
            in_img = cv2.resize(in_img, (args.input_width, args.input_height))
            seg_img = cv2.resize(seg_img  , (input_width , input_height ))

        seg_img = cv2.addWeighted(seg_img,0.7,in_img,0.3,0, dtype=cv2.CV_8UC1)
	#cv2.imwrite(  outName , seg_img )
        if videoout is not None:
	    seg_img = cv2.resize(seg_img  , (w , h ))
            videoout.write(seg_img)
        else:
            cv2.imshow("Prediction", seg_img)
            c = cv2.waitKey(0) & 0x7F
            if c == 27:
                exit()

if videoout is not None:
    videoout.release()
exit()
