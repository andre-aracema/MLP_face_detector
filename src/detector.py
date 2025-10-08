import cv2
import numpy as np
from tensorflow.keras.models import load_model

from .pre_processamento import calcular_iou

class FaceDetector:
    def __init__(self, model_path, tam_janela=(32, 32), confidence_treshould=0.95,
                scala_factor=0.75, stride=6, iou_nms_treshould=0.2):

        self.model = load_model(model_path)
        self.tam_janela = tam_janela
        self.confidence_treshould = confidence_treshould
        self.scala_factor = scala_factor
        self.stride = stride
        self.iou_nms_treshould = iou_nms_treshould

    def sliding_window_multiscale(self, image):
        all_detections = []
        scale = 1.0
        
        h_janela, w_janela = self.tam_janela

        # Copia a imagem para não modificar a original
        image_resized = image.copy()

        while image_resized.shape[0] >= h and image_resized.shape[1] >= w:
            patches = []
            coords = []
            
            # Gera todos os patches para a escala atual
            for y in range(0, image_resized.shape[0] - h_janela, self.stride):
                for x in range(0, image_resized.shape[1] - w_janela, self.stride):
                    patch = image_resized[y:y + h_janela, x:x + w_janela]
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

    def non_max_suression(self, detections):
        if not detections:
            return []

        sorted_boxes = sorted(detections, key=lambda x: x[4], reverse=True)