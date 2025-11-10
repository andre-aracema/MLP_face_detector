"""
Comandos:
  - preprocess: Gera os arquivos .npy (face/non-face) do dataset.
  - train:      Treina o modelo MLP usando os dados .npy (sem estourar a RAM).
  - bootstrap:  Treina utilizando hard negatives.
  - detect:     Detecta faces em uma imagem usando o modelo treinado.
"""

import cv2
import argparse  # Para criar a interface de linha de comando (CLI)
import sys
import os

from src.face_detector import FaceDetector
from src.bootstrap import run_bootstrap_process
from src.training_pipeline import (
    preprocess_and_save_data, 
    run_training_pipeline     
)


# ---------------- Configs de Pré-processamento -------------------------- 


BASE_PATHS = ['data/Derived_YTFaces_160x160/Only_famous_high_quality', 
                 'data/Derived_YTFaces_160x160/Only_famous_low_quality']

NUM_SAMPLES = 1000000
DATA_PATH_1 = 'data/preprocessed/v1_easy_data' 
MODEL_PATH_1 = 'models/detector_v1_easy.keras'
LEARNING_RATE_1 = 0.001   # Adam
EPOCHS_1 = 50


# --------------- Configs de Bootstrap ------------------------------------

DATA_PATH_2 = 'data/preprocessed/v2_hard_data'
MODEL_PATH_2 = 'models/detector_v2_final.keras'
LEARNING_RATE_2 = 0.0002
EPOCHS_2 = 40
HARD_NEGATIVE_THRESHOLD = 0.8


# -------------- Hiperparâmetros ----------------------------------------------

WINDOW_SIZE = (32, 32)
INPUT_SIZE = WINDOW_SIZE[0] * WINDOW_SIZE[1]
BATCH_SIZE = 32

DEFAULT_DETECTION_THRESHOLD = 0.65
DEFAULT_MODEL_FOR_DETECT = MODEL_PATH_2


def main():
    # Cria o "parser" principal que lerá os comandos
    parser = argparse.ArgumentParser(description="Ferramenta de Treinamento e Detecção de Faces com MLP.")
    
    # Cria um "subparser". Permite ter comandos separados
    subparsers = parser.add_subparsers(dest='command', required=True, help='Ação a ser executada')

    # Comando 'preprocess'
    subparsers.add_parser('preprocess', 
                          help='Pré-processa dados "fáceis" (Versão 1).')

    # Comando 'train'
    subparsers.add_parser('train', 
                          help='Treina um modelo (Versão 1 ou Versão 2).')

    # Argumentos específicos que o comando 'train' aceita:
    parser_train.add_argument('--config', type=str, default='v1', choices=['v1', 'v2'],
                              help="Configuração de treino: 'v1' (fácil) ou 'v2' (difícil/HNM). Padrão: v1")

    # Comando 'bootstrap'
    subparsers.add_parser('bootstrap',
                          help='Executa o pipeline de Hard Negative Mining')

    # Comando 'detect'
    parser_detect = subparsers.add_parser('detect', 
                                          help='Detecta faces em uma imagem usando um modelo treinado.')
    
    # Argumentos específicos que o comando 'detect' aceita:
    parser_detect.add_argument('--image', type=str, required=True, 
                               help='Caminho para a imagem de entrada.')
    parser_detect.add_argument('--model', type=str, default=DEFAULT_DETECTION_THRESHOLD, 
                               help=f'Caminho para o modelo .keras (padrão: {DEFAULT_DETECTION_THRESHOLD})')
    parser_detect.add_argument('--threshold', type=float, default=DEFAULT_DETECTION_THRESHOLD, 
                               help=f'Limiar de confiança para detecção (padrão: {DEFAULT_DETECTION_THRESHOLD})')

    # Analisa os argumentos fornecidos pelo usuário
    args = parser.parse_args()

    try:
        if args.command == 'preprocess':
            print("################### MODO: PRÉ-PROCESSAMENTO ###################")
            preprocess_and_save_data(
                base_paths=BASE_PATHS,
                num_samples=NUM_SAMPLES,
                window_size=WINDOW_SIZE,
                save_path_base=DATA_PATH_1
            )

            print("Pré-processamento concluído.")
    
        elif args.command == 'train':
            print("################### MODO: TREINAMENTO (Config: {args.config}) ###################")
            if args.config == 'v1':
                data_path = DATA_PATH_1
                model_path = MODEL_PATH_1
                lr = LEARNING_RATE_1
                epochs = EPOCHS_1
            else: 
                data_path = DATA_PATH_2
                model_path = MODEL_PATH_2
                lr = LEARNING_RATE_2
                epochs = EPOCHS_2
            
            data_file_check = f"{DATA_PATH_1}_face.npy"
            if not os.path.exists(data_file_check):
                print(f"Erro: Dados 1 não encontrados em {data_file_check}. Execute:")
                if args.config == 'v1':
                    print(f"Execute: python {sys.argv[0]} preprocess")
                else:
                    print(f"Execute: python {sys.argv[0]} bootstrap")
                sys.exit(1)
                
            run_training_pipeline(
                data_path_base=data_path,
                window_size=WINDOW_SIZE,
                input_size=INPUT_SIZE,
                model_save_path=model_path,
                learning_rate=lr,    
                epochs=epochs,
                batch_size=BATCH_SIZE
            )

            print(f"Treinamento '{args.config}' concluído. Modelo salvo em: {model_path}")

        elif args.command == 'bootstrap':
            print("################### MODO: BOOTSTRAP ###################")
            data_file_check = f"{DATA_PATH_1}_face.npy"
            if not os.path.exists(data_file_check):
                print(f"Erro: Dados 1 não encontrados em {data_file_check}. Execute primeiro:")
                print(f"python {sys.argv[0]} preprocess")
                sys.exit(1)

            run_bootstrap_process(
                data_path_v1=DATA_PATH_1,
                base_paths_for_mining=BASE_PATHS,
                data_path_v2=DATA_PATH_2,
                model_v1_path=MODEL_PATH_1,
                window_size=WINDOW_SIZE,
                input_size=INPUT_SIZE,
                learn_rate_v1=LEARNING_RATE_1,
                epochs_v1=EPOCHS_1,
                batch_size=BATCH_SIZE,
                hard_negative_threshold=HARD_NEGATIVE_THRESHOLD,
                sample_hnm_to_faces=True
            )

        elif args.command == 'detect':
            print("################### MODO: DETECÇÃO ###################")
            if not os.path.exists(args.model):
                print(f"Erro: Modelo não encontrado em '{args.model}'.")
                print(f"Para treinar o V1 (fácil): python {sys.argv[0]} train --config v1")
                print(f"Para treinar o V2 (difícil): python {sys.argv[0]} train --config v2")
                sys.exit(1)
            
            detector = FaceDetector(
                model_path=args.model,
                confidence_threshold=args.threshold
            )
            
            # Roda a detecção na imagem fornecida
            image_color, detections = detector.detect(args.image)

            if image_color is not None:
                print(f"Encontradas {len(detections)} faces.")

                # Desenha as caixas nas imagens
                image_with_boxes = FaceDetector.draw_boxes(image_color.copy(), detections)

                # Mostra a imagem em uma janela do OpenCV
                cv2.imshow("Faces Detectadas (Pressione qualquer tecla para sair)", image_with_boxes)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Erro: Não foi possível carregar a imagem em '{args.image}'.")

    except Exception as e:
        print(f"\nUM ERRO INESPERADO OCORREU")
        print(f"Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()