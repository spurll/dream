#!/usr/bin/python3

# Adapted from: https://github.com/google/deepdream/blob/master/dream.ipynb

import numpy
import PIL.Image
from scipy import ndimage
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
from io import BytesIO

import caffe

def showarray(a, fmt='jpeg'):
    a = numpy.uint8(numpy.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

net_fn = 'deploy.prototxt'
param_fn = 'bvlc_googlenet.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()

text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

# mean is ImageNet mean, which is training set dependent
# channel_swap is necessary because the reference model is BGR instead of RGB
net = caffe.Classifier(
    'tmp.prototxt', param_fn, mean=numpy.float32([104.0, 116.0, 122.0]),
    channel_swap=(2, 1, 0)
)

# Utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return numpy.float32(numpy.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def postprocess(net, img):
    return numpy.dstack((img + net.transformer.mean['data'])[::-1])

# Making the "dream" images is very simple. Essentially it is just a gradient
# ascent process that tries to maximize the L2 norm of activations of a
# particular DNN layer.

def objective_L2(dst):
    dst.diff[:] = dst.data

def make_step(
    net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True,
    objective=objective_L2
):
    '''
    Basic gradient ascent step, applying a random jitter offset to the image
    and normalizing the magnitude of the gradient ascent steps.
    '''

    src = net.blobs['data'] # Input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = numpy.random.randint(-jitter, jitter + 1, 2)

    # Apply jitter shift
    src.data[0] = numpy.roll(numpy.roll(src.data[0], ox, -1), oy, -2)

    net.forward(end=end)
    objective(dst)          # Specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]

    # Apply normalized ascent step to the input image
    src.data[:] += step_size / numpy.abs(g).mean() * g

    # Unshift image
    src.data[0] = numpy.roll(numpy.roll(src.data[0], -ox, -1), -oy, -2)

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = numpy.clip(src.data, -bias, 255 - bias)

def deepdream(
    net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
    end='inception_4c/output', clip=True, **step_params
):
    '''
    Implement an ascent through different scales. We call these scales
    "octaves".
    '''

    # Prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in range(octave_n - 1):
        octaves.append(
            # ndimage? numpy?
            ndimage.zoom(
                octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale),
                order=1
            )
        )

    src = net.blobs['data']
    # Allocate image for network-produced details
    detail = numpy.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # Upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = ndimage.zoom(
                detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1
            )

        # Resize the network's input image size
        src.reshape(1, 3, h, w)
        src.data[0] = octave_base + detail
        for i in range(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # Visualization
            vis = postprocess(net, src.data[0])
            if not clip:
                # Adjust image contrast if clipping is disabled
                vis = vis * (255.0 / numpy.percentile(vis, 99.98))

            showarray(vis)
            print(octave, i, end, vis.shape)
            clear_output(wait=True)

        # Extract details produced on the current octave
        detail = src.data[0]-octave_base

    # Returning the resulting image
    return postprocess(net, src.data[0])

frame = numpy.float32(PIL.Image.open('sky1280px.jpg'))
# showarray(img)

frame_i = 0
h, w = frame.shape[:2]
s = 0.05                # Scaling coefficient
for i in range(100):
    frame = deepdream(net, frame)
    PIL.Image.fromarray(numpy.uint8(frame)).save("frames/%04d.jpg" % frame_i)
    frame = ndimage.affine_transform(
        frame, [1 - s, 1 - s, 1], [h * s / 2, w * s / 2, 0], order=1
    )
    frame_i += 1
