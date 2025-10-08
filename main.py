import cv2
from src.treinamento import treinar
from src.detector import FaceDetector

def main():
    ACAO_EXECUTAR = 'detectar'
    CAMINHO_IMAGEM = 'data/Derived_YTFaces_160x160/More_people/Aaron_Sorkin/3/3.425.jpg'

    if ACAO_EXECUTAR == 'treinar':
        treinar()

    elif ACAO_EXECUTAR == 'detectar':
        detector = FaceDetector(model_path='models/detector_faces.keras')
        image_color, detections = detector.detectar(CAMINHO_IMAGEM)

        if image_color is not None:
            image_with_boxes = detector.desenhar_caixas(image_color.copy(), detections)
            cv2.imshow("Faces Detectadas", image_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()