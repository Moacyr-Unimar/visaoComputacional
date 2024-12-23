import cv2
from cv2.typing import MatLike
import matplotlib.pyplot as plt


def main() -> None:
    cat_image: MatLike = cv2.imread(
        filename="/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto1.jpg",
        flags=cv2.IMREAD_GRAYSCALE,
    )
    hist = cv2.calcHist(
        images=[cat_image],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
    )
    cat_equalized = cv2.equalizeHist(src=cat_image)
    hist2 = cv2.calcHist(
        images=[cat_equalized],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
    )

    cl_ahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img: MatLike = cl_ahe.apply(src=cat_image)
    hist3: MatLike = cv2.calcHist(
        images=[cl_img],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
    )

    hist = cv2.normalize(hist, None, 0, 1.0, cv2.NORM_MINMAX)
    hist2 = cv2.normalize(hist2, None, 0, 1.0, cv2.NORM_MINMAX)
    hist3 = cv2.normalize(hist3, None, 0, 1.0, cv2.NORM_MINMAX)

    cv2.imwrite(
        filename="/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto1_gs.jpg",
        img=cat_image,
    )
    cv2.imwrite(
        filename="/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto1_eq.jpg",
        img=cat_equalized,
    )
    cv2.imwrite(
        filename="/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto1_clahe.jpg",
        img=cl_img,
    )

    fig, axs = plt.subplots(nrows=3)
    fig.suptitle(t="Histogramas")
    axs[0].plot(hist)
    axs[1].plot(hist2)
    axs[2].plot(hist3)
    plt.show()


if __name__ == "__main__":
    main()
