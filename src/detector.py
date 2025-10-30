import cv2
import numpy as np
from tensorflow.keras.models import load_model

from .pre_processamento import calcular_iou

class FaceDetector:
    def __init__(self, model_path, tam_janela=(32, 32), confidence_threshold=0.995,
            scale_factor=0.9, stride=4, iou_nms_threshold=0.01, variance_threshold=750, edge_density_threshold=0.1):

        self.model = load_model(model_path)
        self.tam_janela = tam_janela
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.stride = stride
        self.iou_nms_threshold = iou_nms_threshold
        self.variance_threshold = variance_threshold
        self.edge_density_threshold = edge_density_threshold

    def sliding_window_multiscale(self, image):
        all_detections = []
        scale = 1.0
        
        h_janela, w_janela = self.tam_janela

        # Copia a imagem para não modificar a original
        image_resized = image.copy()

        while image_resized.shape[0] >= h_janela and image_resized.shape[1] >= w_janela:
            patches = []
            coords = []
            
            # Gera todos os patches para a escala atual
            for y in range(0, image_resized.shape[0] - h_janela, self.stride):
                for x in range(0, image_resized.shape[1] - w_janela, self.stride):
                    patch = image_resized[y:y + h_janela, x:x + w_janela]
                    
                    # verifica a variancia da imagem
                    if patch.var() < self.variance_threshold:
                        continue

                    # verifica as bordas da imagem
                    edge_map = cv2.Canny(patch, 50, 150)
                    density = np.count_nonzero(edge_map) / patch.size
                    if density < self.edge_density_threshold:
                        continue

                    patches.append(patch)
                    coords.append((x, y))

            if not patches:
                break

            patches_np = np.array(patches)
            patches_normalized = patches_np.reshape(len(patches), -1) / 255.0
            scores = self.model.predict(patches_normalized, verbose=0).flatten()

            for i, score in enumerate(scores):
                if score >= self.confidence_threshold:
                    x, y = coords[i]
                    x1_orig = int(x / scale)
                    y1_orig = int(y / scale)
                    x2_orig = int((x + w_janela) / scale)
                    y2_orig = int((y + h_janela) / scale)
                    all_detections.append((x1_orig, y1_orig, x2_orig, y2_orig, score))

            new_width = int(image_resized.shape[1] * self.scale_factor)
            new_height = int(image_resized.shape[0] * self.scale_factor)
            image_resized = cv2.resize(image_resized, (new_width, new_height), interpolation=cv2.INTER_AREA)
            scale *= self.scale_factor

        return all_detections

    def non_max_suppression(self, detections):
        if not detections:
            return []

        sorted_boxes = sorted(detections, key=lambda x: x[4], reverse=True)

        final_boxes = []
        while sorted_boxes:
            chosen_box = sorted_boxes.pop(0)
            final_boxes.append(chosen_box)

            # Remove caixas que têm alta sobreposição com a caixa escolhida
            sorted_boxes = [box for box in sorted_boxes if calcular_iou(chosen_box[:4], box[:4]) < self.iou_nms_threshold]

        return final_boxes

    def detectar(self, image_path):
        image_color = cv2.imread(image_path)
        if image_color is None:
            return None, []
            
        gray_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        
        raw_detections = self.sliding_window_multiscale(gray_image)
        final_detections = self.non_max_suppression(raw_detections)
        
        return image_color, final_detections

    @staticmethod
    def desenhar_caixas(image, detections):
        for (x1, y1, x2, y2, score) in detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            score_text = f"{score:.2f}"
            cv2.putText(image, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image