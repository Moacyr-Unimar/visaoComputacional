import cv2
import matplotlib.pyplot as plt


def main():
    cat_image = cv2.imread('/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto1.jpg')
    hist1 = cv2.calcHist(cat_image, [0], None, [256], [0, 256])
    hist2 = cv2.calcHist(cat_image, [1], None, [256], [0, 256])
    hist3 = cv2.calcHist(cat_image, [2], None, [256], [0, 256])

    plt.title('Histograma P&B')
    plt.xlabel("Intensidade")
    plt.ylabel("Quantidade de Pixels")
    plt.plot(hist1, c='blue')
    plt.plot(hist2, c='green')
    plt.plot(hist3, c='red')
    plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':
    main()
