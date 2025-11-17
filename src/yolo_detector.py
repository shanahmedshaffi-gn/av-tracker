#!/usr/bin/env python3
"""
Unified YOLO Detector Module
============================

Supports both YOLOv5 and YOLOv11 ONNX models with automatic format detection.

YOLOv5: Single output [1, N, 7] with [x_center, y_center, w, h, obj, class0, class1]
YOLOv11: Three outputs - boxes [1, N, 4], scores [1, N], classes [1, N]

Author: Unified YOLO System
Created: November 13, 2025
"""

import cv2
import numpy as np
import onnxruntime as ort
import logging
import platform
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Unified YOLO detector supporting both v5 and v11 formats.
    
    Auto-detects model format from output shapes.
    """
    
    def __init__(self, model_path: str, 
                 confidence_threshold: float = 0.5, 
                 nms_threshold: float = 0.45,
                 use_gpu: bool = False):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to ONNX model
            confidence_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for overlapping boxes
            use_gpu: Whether to attempt GPU acceleration
        """
        logger.info(f"Initializing YOLO detector: {model_path}")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Configure providers
        providers = self._get_optimal_providers(use_gpu)
        
        # Session options for performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        
        logger.info(f"Using providers: {providers}")
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        # Get model details
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.model_width = input_shape[3]
        self.model_height = input_shape[2]
        
        # Detect model format
        outputs = self.session.get_outputs()
        self.num_outputs = len(outputs)
        
        if self.num_outputs == 1:
            self.model_type = "yolov5"
            logger.info(f"Detected YOLOv5 format: {outputs[0].shape}")
        elif self.num_outputs == 3:
            self.model_type = "yolov11"
            logger.info(f"Detected YOLOv11 format: boxes{outputs[0].shape}, scores{outputs[1].shape}, classes{outputs[2].shape}")
        else:
            raise ValueError(f"Unsupported model format with {self.num_outputs} outputs")
        
        logger.info(f"Model: {self.model_type}, Input: {self.model_width}x{self.model_height}")
        logger.info(f"Execution providers: {self.session.get_providers()}")
    
    def _get_optimal_providers(self, use_gpu: bool) -> List[str]:
        """Get optimal execution providers based on platform"""
        available = ort.get_available_providers()
        logger.info(f"Available providers: {available}")
        
        if not use_gpu:
            return ["CPUExecutionProvider"]
        
        if platform.system() == "Darwin":  # macOS
            # CoreML can be finicky with some models
            if "CoreMLExecutionProvider" in available:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
        else:  # Linux/Windows
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
    
    def detect_heads(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect heads in frame and return bounding boxes with confidence
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        if self.model_type == "yolov5":
            return self._detect_yolov5(frame, target_class=0)
        else:
            return self._detect_yolov11(frame, target_class=0)
    
    def detect_bodies(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect bodies in frame and return bounding boxes with confidence
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        if self.model_type == "yolov5":
            return self._detect_yolov5(frame, target_class=1)
        else:
            return self._detect_yolov11(frame, target_class=1)
    
    def detect_heads_and_bodies(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        Detect both heads and bodies in a single pass (more efficient)
        
        Args:
            frame: Input BGR image
            
        Returns:
            (head_boxes, body_boxes) where each is a list of (x1, y1, x2, y2, confidence)
        """
        # Preprocess once
        resized_frame, padding, scale_ratio = self._preprocess_frame(frame)
        model_input = self._frame_to_model_input(resized_frame)
        
        # Run inference once
        outputs = self.session.run(None, {self.input_name: model_input})
        
        # Process both classes
        if self.model_type == "yolov5":
            heads = self._process_yolov5_outputs(outputs, 0, padding, scale_ratio)
            bodies = self._process_yolov5_outputs(outputs, 1, padding, scale_ratio)
        else:
            heads = self._process_yolov11_outputs(outputs, 0, padding, scale_ratio)
            bodies = self._process_yolov11_outputs(outputs, 1, padding, scale_ratio)
        
        return heads, bodies
    
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple, float]:
        """Resize frame with padding to model input size"""
        original_h, original_w = frame.shape[:2]
        ratio = min(self.model_width / original_w, self.model_height / original_h)
        new_w, new_h = int(original_w * ratio), int(original_h * ratio)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        delta_w = self.model_width - new_w
        delta_h = self.model_height - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        padding = (top, bottom, left, right)
        return padded, padding, ratio
    
    def _frame_to_model_input(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to model input format"""
        # Use float32 for YOLOv11 (better performance), float16 for YOLOv5
        dtype = np.float32 if self.model_type == "yolov11" else np.float16
        normalized = frame.astype(dtype) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return batched
    
    def _detect_yolov5(self, frame: np.ndarray, target_class: int) -> List[Tuple[int, int, int, int, float]]:
        """Detect using YOLOv5 format"""
        resized_frame, padding, scale_ratio = self._preprocess_frame(frame)
        model_input = self._frame_to_model_input(resized_frame)
        outputs = self.session.run(None, {self.input_name: model_input})
        return self._process_yolov5_outputs(outputs, target_class, padding, scale_ratio)
    
    def _detect_yolov11(self, frame: np.ndarray, target_class: int) -> List[Tuple[int, int, int, int, float]]:
        """Detect using YOLOv11 format"""
        resized_frame, padding, scale_ratio = self._preprocess_frame(frame)
        model_input = self._frame_to_model_input(resized_frame)
        outputs = self.session.run(None, {self.input_name: model_input})
        return self._process_yolov11_outputs(outputs, target_class, padding, scale_ratio)
    
    def _process_yolov5_outputs(self, outputs, target_class: int, 
                                 padding: Tuple, scale_ratio: float) -> List[Tuple[int, int, int, int, float]]:
        """Process YOLOv5 output format"""
        detections = outputs[0]
        
        # Filter by confidence
        candidates = detections[..., 4] > self.confidence_threshold
        preds = detections[candidates]
        
        if preds.shape[0] == 0:
            return []
        
        # Apply objectness to class scores
        preds[:, 5:] *= preds[:, 4:5]
        
        # Convert boxes from xywh to xyxy
        boxes = self._xywh_to_xyxy(preds[:, :4])
        boxes = np.clip(boxes, 0, None)
        
        # Filter by target class
        class_scores = preds[:, 5 + target_class]
        indices = class_scores > self.confidence_threshold
        class_boxes = boxes[indices]
        class_scores = class_scores[indices]
        
        if len(class_scores) == 0:
            return []
        
        # Apply NMS
        nms_indices = cv2.dnn.NMSBoxes(
            class_boxes.tolist(),
            class_scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Adjust coordinates back to original frame
        top, _, left, _ = padding
        bboxes = []
        
        for i in nms_indices.flatten():
            x1, y1, x2, y2 = class_boxes[i]
            conf = class_scores[i]
            
            # Remove padding and scale
            x1 = (x1 - left) / scale_ratio
            y1 = (y1 - top) / scale_ratio
            x2 = (x2 - left) / scale_ratio
            y2 = (y2 - top) / scale_ratio
            
            bboxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        
        return bboxes
    
    def _process_yolov11_outputs(self, outputs, target_class: int,
                                  padding: Tuple, scale_ratio: float) -> List[Tuple[int, int, int, int, float]]:
        """Process YOLOv11 output format"""
        boxes = outputs[0][0]    # [N, 4] - [y1, x1, y2, x2] (swapped!)
        scores = outputs[1][0]   # [N]
        classes = outputs[2][0]  # [N]
        
        # Filter by class
        class_mask = classes == target_class
        filtered_boxes = boxes[class_mask]
        filtered_scores = scores[class_mask]
        
        # Filter by confidence
        conf_mask = filtered_scores > self.confidence_threshold
        filtered_boxes = filtered_boxes[conf_mask]
        filtered_scores = filtered_scores[conf_mask]
        
        if len(filtered_scores) == 0:
            return []
        
        # Convert from [y1, x1, y2, x2] to [x1, y1, x2, y2]
        swapped_boxes = np.zeros_like(filtered_boxes)
        swapped_boxes[:, 0] = filtered_boxes[:, 1]  # x1
        swapped_boxes[:, 1] = filtered_boxes[:, 0]  # y1
        swapped_boxes[:, 2] = filtered_boxes[:, 3]  # x2
        swapped_boxes[:, 3] = filtered_boxes[:, 2]  # y2
        
        boxes_xyxy = np.clip(swapped_boxes, 0, None)
        
        # Apply NMS
        nms_indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            filtered_scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Adjust coordinates back to original frame
        top, _, left, _ = padding
        bboxes = []
        
        for i in nms_indices.flatten():
            x1, y1, x2, y2 = boxes_xyxy[i]
            conf = filtered_scores[i]
            
            # Remove padding and scale
            x1 = (x1 - left) / scale_ratio
            y1 = (y1 - top) / scale_ratio
            x2 = (x2 - left) / scale_ratio
            y2 = (y2 - top) / scale_ratio
            
            bboxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        
        return bboxes
    
    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]"""
        result = np.zeros_like(boxes)
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return result
