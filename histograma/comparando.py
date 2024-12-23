from typing import Any
import cv2
import numpy as np

# from cv2.typing import MatLike
# import matplotlib.pyplot as plt


def main() -> None:
    imagens: list[str] = [
        "/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto1.jpg",
        "/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto2.jpg",
        "/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto3.jpg",
        "/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto4.jpg",
        "/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto5.jpg",
        "/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/foto6.jpg",
        "/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/histograma/images/gatinho.png",
    ]
    results: dict[Any, Any] = {}
    data_frame_result: np.ndarray[np.float64] = np.zeros(
        shape=(len(imagens), len(imagens)),
    )
    for idx_gal, i in enumerate(iterable=imagens):
        results[i] = []
        image_bank = cv2.imread(
            filename=i,
            flags=cv2.IMREAD_GRAYSCALE,
        )
        cl_ahe: cv2.CLAHE = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        cl_img = cl_ahe.apply(src=image_bank)
        cl_img = cv2.calcHist(
            images=[cl_img],
            channels=[0],
            mask=None,
            histSize=[256],
            ranges=[0, 256],
        )
        cl_img = cv2.normalize(
            src=cl_img,
            dst=cl_img,
            alpha=0,
            beta=1.0,
            norm_type=cv2.NORM_MINMAX,
        )
        for idx_probe, j in enumerate(iterable=imagens):
            image_probe = cv2.imread(
                filename=j,
                flags=cv2.IMREAD_GRAYSCALE,
            )
            cl_ahe_p: cv2.CLAHE = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            )
            cl_img_p = cl_ahe_p.apply(src=image_probe)
            cl_img_p = cv2.calcHist(
                images=[cl_img_p],
                channels=[0],
                mask=None,
                histSize=[256],
                ranges=[0, 256],
            )
            cl_img_p = cv2.normalize(
                src=cl_img_p,
                dst=cl_img_p,
                alpha=0,
                beta=1.0,
                norm_type=cv2.NORM_MINMAX,
            )
            data_frame_result[idx_gal][idx_probe] = cv2.compareHist(
                H1=cl_img,
                H2=cl_img_p,
                method=1,
            )
    ful_image = None
    for idx in range(len(imagens)):
        idxs_vals = data_frame_result[idx].argsort()
        curr_img = cv2.imread(filename=imagens[idx])
        curr_img = cv2.resize(
            src=curr_img,
            dsize=(250, 250),
        )
        matched_img = cv2.imread(filename=imagens[idxs_vals[1]])
        matched_img = cv2.resize(
            src=matched_img,
            dsize=(250, 250),
        )
        f_image = np.concatenate(
            (curr_img, matched_img),
            axis=0,
        )
        if ful_image is not None:
            ful_image = np.concatenate(
                (ful_image, f_image),
                axis=1,
            )
        else:
            ful_image = f_image
    cv2.imshow(
        "Janela da Imagem",
        ful_image,
    )
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
