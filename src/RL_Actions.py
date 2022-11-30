import cv2
import numpy as np
import matplotlib.pyplot as plt

import DehazingDarkChannelPrior.dehaze as dcp
import DehazingColorAttenuation.dehaze as ca

def CE(src):
    image_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    return image_rgb

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

def ChannelValue(img,val,channel):
    img_channel=img[:,:,channel]
    img_channel[img_channel*val<255]=img_channel[img_channel*val<255]*1.05
    img_channel[img_channel*val>=255]=255
    img[:,:,channel]=img_channel[:]
    return img

def dehazeDCP(src):
    I = src.astype("float64") / 255
    # normalizing the data to 0 - 1 (Since 255 is the maximum value)

    dark = dcp.DarkChannel(I, 15)
    # extracting dark channel prior
    A = dcp.AtmLight(I, dark)
    # extracting global atmospheric lighting
    # Transmission is an estimate of how much of the light from the
    # original object is making it through the haze at each pixel
    te = dcp.TransmissionEstimate(I, A, 15)
    t = dcp.TransmissionRefine(src, te)
    # atmospheric light is subtracted from each pixel in proportion to the transmission at that pixel.
    J = dcp.Recover(I, t, A, 0.1)

    return (J*255).astype(np.uint8)

def dehazeCA(src):
    # Read the Image
    # _I = cv2.imread(fn)
    # opencv reads any image in Blue-Green-Red(BGR) format,
    # so change it to RGB format, which is popular.
    I = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    # Split Image to Hue-Saturation-Value(HSV) format.
    H, S, V = cv2.split(cv2.cvtColor(src, cv2.COLOR_BGR2HSV))
    V = V / 255.0
    S = S / 255.0

    # Calculating Depth Map using the linear model fit by ZHU et al.
    # Refer Eq(8) in mentioned research paper
    # Values given under EXPERIMENTS section
    theta_0 = 0.121779
    theta_1 = 0.959710
    theta_2 = -0.780245
    sigma = 0.041337
    epsilon = np.random.normal(0, sigma, H.shape)
    D = theta_0 + theta_1 * V + theta_2 * S + epsilon

    # saving depth map
    # plt.imsave(os.path.join(filepath, filename + "_depth_map.jpg"), D)

    # Local Minima of Depth map
    LMD = ca.localmin(D, 15)
    # LMD = D

    # Guided Filtering
    r = 8
    # try r=2, 4, 8 or 18
    eps = 0.2 * 0.2
    # try eps=0.1^2, 0.2^2, 0.4^2
    # eps *= 255 * 255;   # Because the intensity range of our images is [0, 255]
    GD = ca.guide(D, LMD, r, eps)

    J = ca.postprocessing(GD, I,V)

    return (J*255).astype(np.uint8)

