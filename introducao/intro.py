import cv2
from cv2.typing import MatLike


def main() -> None:
    cat_image: MatLike = cv2.imread(
        filename="/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/introducao/images/gatinho.png",
        flags=cv2.IMREAD_GRAYSCALE,
    )

    # inicioRetangulo = (20,10)
    # fimRetangulo = (100,90)
    # cor = (0,0,255)
    # grossura = 5
    # cat_image = cv2.rectangle(cat_image,inicioRetangulo,fimRetangulo,cor,grossura)

    cv2.imshow(
        winname="Janela da imagem",
        mat=cat_image,
    )
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()

    cv2.imwrite(
        filename="/Users/moacyrfc/projects/Unimar/VisaoComputacional/professor/introducao/images/gatinhoGray.jpg",
        img=cat_image,
    )


if __name__ == "__main__":
    main()
