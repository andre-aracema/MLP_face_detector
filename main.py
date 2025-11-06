"""
Comandos:
  - preprocess-v1:     Gera dados .npy para o treino inicial (v1).
  - preprocess-mining: Gera dados .npy para a mineração (não-faces).
  - train-v1:          Treina o modelo v1 usando dados v1 pré-processados.
  - bootstrap:         Executa o Hard Negative Mining (cria v1 e v2).
  - detect:            Detecta faces em uma imagem usando um modelo.
"""

import cv2
import argparse
import sys
import os

from src.training_pipeline import (
    preprocess_and_save_data, 
    load_preprocessed_data, 
    run_training_pipeline
)
from src.bootstrap import run_bootstrap_process
from src.face_detector import FaceDetector


# Configs de Pré-processamento e Treino v1
BASE_PATHS_V1 = ['data/Derived_YTFaces_160x160/Only_famous_high_quality', 
                 'data/Derived_YTFaces_160x160/Only_famous_low_quality']
NUM_SAMPLES_V1 = 500000
DATA_PATH_V1 = 'data/preprocessed_v1' 
MODEL_SAVE_PATH_V1 = 'models/detector_faces_v1.keras'

# Configs de Mineração 
BASE_PATHS_MINING = ['data/Derived_YTFaces_160x160/Only_famous_low_quality'] 
NUM_SAMPLES_MINING = 100000
DATA_PATH_MINING = 'data/preprocessed_mining' 
MODEL_SAVE_PATH_V2 = 'models/detector_faces_v2.keras'

# Configs Comuns
WINDOW_SIZE = (32, 32)
INPUT_SIZE = WINDOW_SIZE[0] * WINDOW_SIZE[1]
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
HARD_NEGATIVE_THRESHOLD = 0.99
DEFAULT_DETECTION_THRESHOLD = 0.995


def main():
    parser = argparse.ArgumentParser(description="Ferramenta de Treinamento e Detecção de Faces com MLP.")
    
    # --- Definição dos Comandos ---
    subparsers = parser.add_subparsers(dest='command', required=True, help='Ação a ser executada')

    # Comando 'preprocess-v1'
    subparsers.add_parser('preprocess-v1', 
                          help='Executa o pré-processamento dos dados de treino v1.')
    
    # Comando 'preprocess-mining'
    subparsers.add_parser('preprocess-mining', 
                          help='Executa o pré-processamento dos dados de mineração (não-faces).')

    # Comando 'train-v1'
    subparsers.add_parser('train-v1', 
                          help='Treina o modelo v1 usando dados v1 pré-processados.')

    # Comando 'bootstrap'
    subparsers.add_parser('bootstrap', 
                          help='Executa o pipeline completo de Hard Negative Mining (cria v1 e v2).')

    # Comando 'detect' (seu 'detectar')
    parser_detect = subparsers.add_parser('detect', 
                                          help='Detecta faces em uma imagem usando um modelo treinado.')
    parser_detect.add_argument('--image', type=str, required=True, 
                               help='Caminho para a imagem de entrada.')
    parser_detect.add_argument('--model', type=str, default=MODEL_SAVE_PATH_V2, 
                               help=f'Caminho para o modelo .keras (padrão: {MODEL_SAVE_PATH_V2})')
    parser_detect.add_argument('--threshold', type=float, default=DEFAULT_DETECTION_THRESHOLD, 
                               help=f'Limiar de confiança para detecção (padrão: {DEFAULT_DETECTION_THRESHOLD})')

    args = parser.parse_args()

    try:
        if args.command == 'preprocess-v1':
            print("--- MODO: PRÉ-PROCESSAMENTO (Dados v1) ---")
            preprocess_and_save_data(
                base_paths=BASE_PATHS_V1,
                num_samples=NUM_SAMPLES_V1,
                window_size=WINDOW_SIZE,
                save_path_base=DATA_PATH_V1
            )

        elif args.command == 'preprocess-mining':
            print("--- MODO: PRÉ-PROCESSAMENTO (Dados de Mineração) ---")
            preprocess_and_save_data(
                base_paths=BASE_PATHS_MINING,
                num_samples=NUM_SAMPLES_MINING,
                window_size=WINDOW_SIZE,
                save_path_base=DATA_PATH_MINING
            )

        elif args.command == 'train-v1':
            print("--- MODO: TREINAMENTO (v1) ---")
            if not os.path.exists(f"{DATA_PATH_V1}_face.npy"):
                print(f"Erro: Dados v1 não encontrados. Execute:")
                print(f"python {sys.argv[0]} preprocess-v1")
                sys.exit(1)
                
            X, y, _ = load_preprocessed_data(DATA_PATH_V1, WINDOW_SIZE)
            run_training_pipeline(
                X_data=X, y_data=y, input_size=INPUT_SIZE,
                model_save_path=MODEL_SAVE_PATH_V1, learning_rate=LEARNING_RATE,
                epochs=EPOCHS, batch_size=BATCH_SIZE
            )

        elif args.command == 'bootstrap':
            print("--- MODO: BOOTSTRAP (HNM) ---")
            # Verifica se os dados de v1 E de mineração existem
            if not (os.path.exists(f"{DATA_PATH_V1}_face.npy") and 
                    os.path.exists(f"{DATA_PATH_MINING}_non_face.npy")):
                print("Erro: Dados pré-processados não encontrados.")
                if not os.path.exists(f"{DATA_PATH_V1}_face.npy"):
                    print(f"Execute: python {sys.argv[0]} preprocess-v1")
                if not os.path.exists(f"{DATA_PATH_MINING}_non_face.npy"):
                    print(f"Execute: python {sys.argv[0]} preprocess-mining")
                sys.exit(1)
            
            # Chama o módulo de bootstrap (que é inteligente e pula o treino do v1 se o modelo já existir)
            run_bootstrap_process(
                data_path_v1=DATA_PATH_V1,
                data_path_mining=DATA_PATH_MINING,
                model_v1_path=MODEL_SAVE_PATH_V1,
                model_v2_path=MODEL_SAVE_PATH_V2,
                window_size=WINDOW_SIZE,
                input_size=INPUT_SIZE,
                learn_rate=LEARNING_RATE,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                hard_negative_threshold=HARD_NEGATIVE_THRESHOLD
            )

        elif args.command == 'detect':
            print(f"--- MODO: DETECÇÃO (Modelo: {args.model}) ---")
            if not os.path.exists(args.model):
                print(f"Erro: Modelo não encontrado em '{args.model}'.")
                print(f"Você pode treiná-lo executando: python {sys.argv[0]} bootstrap")
                sys.exit(1)
            
            detector = FaceDetector(
                model_path=args.model,
                confidence_threshold=args.threshold
            )
            
            image_color, detections = detector.detect(args.image)

            if image_color is not None:
                print(f"Encontradas {len(detections)} faces.")
                image_with_boxes = FaceDetector.draw_boxes(image_color.copy(), detections)
                cv2.imshow("Faces Detectadas (Pressione qualquer tecla para sair)", image_with_boxes)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Erro: Não foi possível carregar a imagem em '{args.image}'.")

    except Exception as e:
        print(f"\n--- UM ERRO INESPERADO OCORREU ---")
        print(f"Erro: {e}")
        # import traceback
        # traceback.print_exc() # Descomente esta linha para depuração detalhada
        sys.exit(1)

if __name__ == "__main__":
    main()