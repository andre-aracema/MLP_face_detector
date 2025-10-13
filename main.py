import cv2
import argparse
from src.treinamento import treinar, pre_processar_salvar
from src.detector import FaceDetector

def main():
    parser = argparse.ArgumentParser(description="Ferramenta de Treinamento e Detecção de Faces com MLP.")
    
    # Cria sub-comandos para cada ação
    subparsers = parser.add_subparsers(dest='acao', required=True, help='Ação a ser executada')

    # Comando 'preprocessar'
    parser_preprocessar = subparsers.add_parser('preprocessar', help='Gera e salva as amostras de face e não-face.')

    # Comando 'treinar'
    parser_treinar = subparsers.add_parser('treinar', help='Executa o pipeline de treinamento inicial do modelo.')

    # Comando 'detectar'
    parser_detectar = subparsers.add_parser('detectar', help='Detecta faces em uma imagem usando um modelo treinado.')
    parser_detectar.add_argument('--imagem', type=str, required=True, help='Caminho para a imagem de entrada.') 
    parser_detectar.add_argument('--modelo', type=str, default='models/detector_faces.keras', 
                                 help='Caminho para o arquivo do modelo .keras. (Ex: models/detector_faces_v2.keras)')

    args = parser.parse_args()

    if args.acao == 'preprocessar':
        print("Iniciando Pré-Processamento (Geração de Dados)")
        pre_processar_salvar()
        print("Pré-Processamento Concluído")

    elif args.acao == 'treinar':
        print("Iniciando Treinamento do Modelo")
        treinar()
        print("Treinamento Concluído")

    elif args.acao == 'detectar':
        print(f"Detectando faces em '{args.imagem}' usando o modelo '{args.modelo}' ")
        
        # O FaceDetector é robusto e está no src/detector.py
        detector = FaceDetector(model_path=args.modelo)
        image_color, detections = detector.detectar(args.imagem)

        if image_color is not None:
            print(f"Encontradas {len(detections)} faces.")
            image_with_boxes = detector.desenhar_caixas(image_color.copy(), detections)
            
            cv2.imshow("Faces Detectadas", image_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Erro: Não foi possível carregar a imagem em '{args.imagem}'.")

if __name__ == "__main__":
    main()