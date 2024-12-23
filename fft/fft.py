import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    imagens = [
        '/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/fft/images/texto1.png',
        '/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/fft/images/texto2.png'
    ]
    for i in imagens:
        imTexto = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        f = np.fft.fft2(imTexto)
        fShift = np.fft.fftshift(f)
        magEspec = 20*np.log(np.abs(fShift))
        magEspec[magEspec < 240] = 0
        magEspec[magEspec >= 240] = 255
        metadeWdt = magEspec.shape[1] // 2
        metadeAlt = magEspec.shape[0] // 2
        if (np.sum(magEspec[:metadeAlt-(metadeAlt//2), metadeWdt-10:metadeWdt+10]) > 1000):
            print('Imagem %s está coom o texto reto' % (i))
        else:
            print('Imagem %s está coom o texto inclinado' % (i))

    '''
    f = np.fft.fft2(imTexto1)
    fShift = np.fft.fftshift(f)
    magEspec = 20*np.log(np.abs(fShift))
    magEspec[magEspec < 240] = 0
    magEspec[magEspec >= 240] = 255

    metadeWdt = magEspec.shape[1] // 2
    metadeAlt = magEspec.shape[0] // 2
    if (np.sum(magEspec[:metadeAlt-(metadeAlt//2),metadeWdt-10:metadeWdt+10]) > 1000):
        print('reto')
    else:
        print('inclinado')

    plt.subplot(221),plt.imshow(imTexto1, cmap = 'gray')
    plt.title('Imagem de entrada'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(magEspec1, cmap = 'gray')
    plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(imTexto2, cmap = 'gray')
    plt.title('Imagem de entrada'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(magEspec2, cmap = 'gray')
    plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    plt.show()
    '''


if __name__ == '__main__':
    main()
