import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import argparse
import Models , LoadBatches
import glob
import itertools
import cv2
import numpy as np

def getSegmentationArr( path , nClasses ):

    try:
        img = cv2.imread(path, 1)
        img = img[:, : , 0]
        h,w = img.shape
        seg_labels = np.zeros((  h , w  , nClasses ))

        for c in range(nClasses):
            seg_labels[: , : , c ] = (img == c ).astype(int)

    except Exception, e:
        print e

    h,w = (h//16)*16, (w//16)*16
    seg_labels = seg_labels[0:h, 0:w]
    seg_labels = np.rollaxis(seg_labels, 2, 0)
    #seg_labels = np.reshape(seg_labels, ( nClasses ))
    return seg_labels


def getImageArr( path , imgNorm="sub_mean" , odering='channels_first' ):

    try:
        img = cv2.imread(path, 1)
        h,w,_ = img.shape
        h,w = (h//16)*16, (w//16)*16

        if imgNorm == "sub_and_divide":
            #img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
            img = np.float32(img) / 127.5 - 1
        elif imgNorm == "sub_mean":
            #img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
        elif imgNorm == "divide":
            #img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img = img/255.0

        img = img[0:h, 0:w,:]
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)

        return img
    except Exception, e:
        print path , e
        #img = np.zeros((  height , width  , 3 ))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img

def generate_batch(images_path, segs_path, batch_size, n_classes):
    images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
    images.sort()
    segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
    segmentations.sort()

    assert len( images ) == len(segmentations)
    zipped = itertools.cycle( zip(images,segmentations) )

    while True:
        X = []
        Y = []
        for _ in range( batch_size) :
            im , seg = zipped.next()
            X.append( getImageArr(im) )
            Y.append( getSegmentationArr( seg , n_classes )  )

        yield np.array(X) , np.array(Y)


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 1 )
parser.add_argument("--val_batch_size", type = int, default = 1 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['accuracy'])


if len( load_weights ) > 0:
	m.load_weights(load_weights)


print "Model output shape" ,  m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth

#G = generate_batch(train_images_path, train_segs_path, train_batch_size, n_classes)

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


#'''
X,Y = G.next()
Y = np.reshape(Y, (output_height, output_width, n_classes))
print("Y shape: ", Y.shape, " max Y: ", np.max(Y))
for i in range(n_classes):
    print("Plane %d sum: %d"%(i, np.sum(Y[:,:,i])))
print("All Plane sum: ", np.sum(Y))
#cv2.imshow("Y0", Y[:,:,0])
#cv2.imshow("Y1", Y[:,:,1])
#c = cv2.waitKey(0) & 0x7F
#if c == 27 or c == ord('q'):
    #exit()

#import pdb; pdb.set_trace()
#'''

if validate:
        #G2 = generate_batch(val_images_path, val_segs_path, val_batch_size, n_classes)
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

for ep in range( epochs ):
    if not validate:
	m.fit_generator( G , 512  , epochs=1 )
    else:
	m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 ,  epochs=1 )

    print("Epoch: %d"%(ep))
    m.save_weights( save_weights_path + "." + str( ep ) )
    m.save( save_weights_path + ".model." + str( ep ) )
