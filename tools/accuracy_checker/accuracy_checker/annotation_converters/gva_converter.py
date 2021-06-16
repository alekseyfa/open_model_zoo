import numpy as np
import os
import json
from .format_converter import BaseFormatConverter, ConverterReturn
from ..utils import read_txt, check_file_existence
from ..config import PathField, BoolField
from ..representation import DetectionAnnotation


class GVAConverter(BaseFormatConverter):
    __provider__ = 'gva'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'images_dir': PathField(optional=True, is_directory=True, description='Images directory'),
            'json_file': PathField(optional=False, is_directory=True, description='Json file'),
            'has_background': BoolField(optional=True, default=False, description='Indicator of background'),
            'add_background_to_label_id': BoolField(
                optional=True, default=False, description='Indicator that need shift labels'
            )
        })
        return params

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.json_file = self.get_value_from_config('json_file')
        self.has_background = self.get_value_from_config('has_background')
        self.shift_labels = self.get_value_from_config('add_background_to_label_id')
        self.raw_detections = self.get_value_from_config('raw_detections')

    def getImagePathById(self, image_id, prefix="", sufix=".jpg"):
        # "000257"
        im_id_str = str(image_id)
        zeros_str = '0' * (6 - len(im_id_str))
        im_id_str = zeros_str + im_id_str
        image_path = prefix + im_id_str + sufix
        return image_path

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = None if not check_content else []
        annotations = list()
        current_img_id = 1
        # FIXME: remove hardcode
        gt_data = self._extract_gt_from_json_file(os.path.join(self.json_file, 'MOT-17-09-SDP_detect_only_gt.json'))
        num_iterations = len(gt_data)
        print("num_iterations: {}".format(num_iterations))
        for video_frame in gt_data:
            objects = video_frame.get('objects', [])
            x_mins, y_mins, x_maxs, y_maxs, scores, labels = [], [], [], [], [], []
            for obj in objects:
                det = obj['detection']
                scores.append(det['confidence'])
                labels.append(int(det['label_id']))
                if not self.raw_detections:
                    x_min, y_min, x_max, y_max = obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']
                else:
                    bbox = det['bounding_box']
                    x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
                x_mins.append(float(x_min))
                y_mins.append(float(y_min))
                x_maxs.append(float(x_max))
                y_maxs.append(float(y_max))
            annotations.append(DetectionAnnotation(self.getImagePathById(current_img_id), np.array(labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs)))
            current_img_id += 1

            if progress_callback and current_img_id % progress_interval == 0:
                progress_callback(current_img_id * 100 / num_iterations)

        return ConverterReturn(annotations, self.generate_meta(), content_errors)

    def _extract_gt_from_json_file(self, file_path: str) -> list():
        if not os.path.exists(file_path):
            raise FileExistsError("The {} path to ground truth json file not found".format(file_path))
        target_parsed = list()
        with open(file_path, 'r') as gt_file:
            target_parsed = json.load(gt_file)

        return target_parsed


    def generate_meta(self):
        labels = ['object']
        label_map = dict()
        for idx, label_name in enumerate(labels):
            label_map[idx + self.has_background] = label_name
        meta = {'label_map': label_map}
        if self.has_background:
            meta['label_map'][0] = 'background'
            meta['background_label'] = 0
        return meta
