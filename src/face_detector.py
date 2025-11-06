import cv2
import numpy as np
from tensorflow.keras.models import load_model, Model
from typing import List, Tuple, Generator

from .preprocessing import calculate_iou

# Uma classe para detectar faces em imagens usando um modelo Keras e uma abordagem de sliding window multiescala
class FaceDetector:

    def __init__(self, model_path: str,
                 window_size: Tuple[int, int] = (32, 32),
                 confidence_threshold: float = 0.995,
                 scale_factor: float = 0.9,
                 stride: int = 4,
                 iou_nms_threshold: float = 0.01,
                 variance_threshold: float = 750.0,
                 edge_density_threshold: float = 0.1):

        self.model: Model = load_model(model_path)
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.stride = stride
        self.iou_nms_threshold = iou_nms_threshold
        self.variance_threshold = variance_threshold
        self.edge_density_threshold = edge_density_threshold

    # Carrega uma imagem e retorna as detecções finais.
    def detect(self, image_path: str) -> Tuple[np.ndarray | None, List[tuple]]:
        
        image_color = cv2.imread(image_path)
        if image_color is None:
            print(f"Erro: Não foi possível carregar a imagem de {image_path}")
            return None, []
            
        gray_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        
        raw_detections = self._sliding_window_multiscale(gray_image)
        final_detections = self._non_max_suppression(raw_detections)
        
        return image_color, final_detections

    @staticmethod
    # Método estático utilitário para desenhar caixas de detecção
    def draw_boxes(image: np.ndarray, detections: List[tuple]) -> np.ndarray:
        
        for (x1, y1, x2, y2, score) in detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            score_text = f"{score:.2f}"
            cv2.putText(image, score_text, (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image

    # Recebe um lote de patches JÁ PROCESSADOS (1D, normalizados, ex: (N, 1024)) e retorna as pontuações brutas do modelo.
    def predict_processed_patches(self, patches_processed: np.ndarray) -> np.ndarray:
        
        if patches_processed.size == 0:
            return np.array([])
        
        return self.model.predict(
            patches_processed, 
            batch_size=256, 
            verbose=1
        ).flatten()

    """-------------------------Funções Auxiliáres-----------------------------"""

    # Cria um gerador para a pirâmide de imagens
    def _build_image_pyramid(self, image: np.ndarray) -> Generator:
        
        scale = 1.0
        h_win, w_win = self.window_size
        current_image = image.copy()

        while current_image.shape[0] >= h_win and current_image.shape[1] >= w_win:
            yield current_image, scale
            
            # Redimensiona para a próxima escala
            new_width = int(current_image.shape[1] * self.scale_factor)
            new_height = int(current_image.shape[0] * self.scale_factor)
            current_image = cv2.resize(current_image, (new_width, new_height), 
                                       interpolation=cv2.INTER_AREA)
            scale *= self.scale_factor

    # Verifica se um patch é interessante o suficiente para ser enviado à rede neural
    def _is_patch_valid(self, patch: np.ndarray) -> bool:
       
        # Filtro de Variância (evita áreas "planas")
        if patch.var() < self.variance_threshold:
            return False

        # Filtro de Densidade de Borda (evita áreas muito simples)
        edge_map = cv2.Canny(patch, 50, 150)
        density = np.count_nonzero(edge_map) / patch.size
        if density < self.edge_density_threshold:
            return False
            
        return True

    # Executa a "janela deslizante" (sliding window) em uma única escala. Retorna apenas os patches que passam nos filtros
    def _generate_and_filter_patches(self, image_scaled: np.ndarray) -> Tuple[List, List]:
        
        h_win, w_win = self.window_size
        valid_patches = []
        valid_coords = []

        for y in range(0, image_scaled.shape[0] - h_win, self.stride):
            for x in range(0, image_scaled.shape[1] - w_win, self.stride):
                patch = image_scaled[y:y + h_win, x:x + w_win]
                
                if self._is_patch_valid(patch):
                    valid_patches.append(patch)
                    valid_coords.append((x, y))
                    
        return valid_patches, valid_coords

    # Converte a lista de patches em array
    def _batch_predict(self, patches: List[np.ndarray]) -> np.ndarray:
        
        if not patches:
            return np.array([])
            
        patches_np = np.array(patches)
        input_size = self.window_size[0] * self.window_size[1]
        patches_processed = patches_np.reshape(len(patches_np), input_size).astype(np.float32) / 255.0
        
        # verbose=0 é para velocidade durante o sliding window
        return self.model.predict(patches_processed, verbose=0).flatten()

    # 1. Gera patches, 2. Prevê, 3. Converte coordenadas
    def _scan_image_scale(self, image_scaled: np.ndarray, scale: float) -> List[tuple]:
        
        h_win, w_win = self.window_size
        patches, coords = self._generate_and_filter_patches(image_scaled)
        
        if not patches:
            return []

        scores = self._batch_predict(patches)
        
        detections = []
        for i, score in enumerate(scores):
            if score >= self.confidence_threshold:
                x, y = coords[i]
                
                # Converte coordenadas de volta para a escala original
                x1_orig = int(x / scale)
                y1_orig = int(y / scale)
                x2_orig = int((x + w_win) / scale)
                y2_orig = int((y + h_win) / scale)
                
                detections.append((x1_orig, y1_orig, x2_orig, y2_orig, score))
                
        return detections

    # Orquestra a pirâmide de imagens e o escaneamento em cada escala
    def _sliding_window_multiscale(self, image: np.ndarray) -> List[tuple]:
        
        all_detections = []
        
        for image_scaled, scale in self._build_image_pyramid(image):
            detections_at_scale = self._scan_image_scale(image_scaled, scale)
            all_detections.extend(detections_at_scale)
            
        return all_detections

    #Limpa as detecções, removendo caixas sobrepostas para a mesma face, mantendo apenas a de maior pontuação
    def _non_max_suppression(self, detections: List[tuple]) -> List[tuple]:
        
        if not detections:
            return []

        # Ordena pela pontuação (score) [índice 4], da maior para a menor
        sorted_boxes = sorted(detections, key=lambda x: x[4], reverse=True)

        final_boxes = []
        while sorted_boxes:
            # Pega a melhor caixa e a remove da lista
            chosen_box = sorted_boxes.pop(0)
            final_boxes.append(chosen_box)

            # Filtra a lista, mantendo apenas caixas que NÃO se sobrepõem muito com a caixa escolhida
            sorted_boxes = [box for box in sorted_boxes 
                            if calculate_iou(chosen_box[:4], box[:4]) < self.iou_nms_threshold]

        return final_boxes