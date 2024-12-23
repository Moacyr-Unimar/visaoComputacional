import cv2
import matplotlib.pyplot as plt
import numpy as np


def detectarBorramento(image, size=60, t=10):
    (ht, wd) = image.shape
    (cX, cY) = (wd // 2, ht // 2)
    fr = np.fft.fft2(image)
    ftShift = np.fft.fftshift(fr)
    ftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(ftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return (mean, mean <= t)


def main():
    imagens = [
        '/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/fft/images/foto3.jpg',
        '/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/fft/images/foto4.jpg'
    ]
    for i in imagens:
        orig = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        media, result = detectarBorramento(orig, t=10)
        print("Foto %s está %s" % (i, 'Borrada' if result else 'Nítida'))


if __name__ == '__main__':
    main()
