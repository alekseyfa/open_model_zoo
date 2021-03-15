import numpy as np
import os
import re
from .format_converter import BaseFormatConverter, ConverterReturn
from ..utils import read_txt, check_file_existence
from ..config import PathField, BoolField
from ..representation import DetectionAnnotation


class MOTConverter(BaseFormatConverter):
    __provider__ = 'mot'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'dataset_dir': PathField(optional=False, is_directory=True, description='Dataset directory'),
            'has_background': BoolField(optional=True, default=False, description='Indicator of background'),
            'add_background_to_label_id': BoolField(
                optional=True, default=False, description='Indicator that need shift labels'
            )
        })
        return params

    def configure(self):
        self.dataset_dir = self.get_value_from_config('dataset_dir')
        self.has_background = self.get_value_from_config('has_background')
        self.shift_labels = self.get_value_from_config('add_background_to_label_id')
        self.global_idx = 0

    def getImagePathById(self, image_id, prefix="", sufix=".jpg"):
        # "000257"
        im_id_str = str(image_id)
        zeros_str = '0' * (6 - len(im_id_str))
        im_id_str = zeros_str + im_id_str
        image_path = prefix + im_id_str + sufix
        return image_path

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation_files = list()
        content_errors = None if not check_content else []
        annotations = list()
        for subdir, dirs, files in os.walk(self.dataset_dir):
            for folder in dirs:
                if folder.endswith('SDP'):
                    annotation_path = os.path.join(subdir, folder, 'det', 'det.txt')
                    annotation_files.append(annotation_path)
                    # print("annotation_path: {}".format(annotation_path))
                    # annotations.extend(self.parse_annotation(annotation_path))
        
        annotation_files.sort(key=lambda x:int(re.search(r'MOT17-(.*?)-SDP', x).group(1)))
        annotation_files = self.remove_invalid_datasets(annotation_files)
        print(annotation_files)
        for annotation_file in annotation_files:
            annotations.extend(self.parse_annotation(annotation_file))
            # if progress_callback and idx % progress_interval == 0:
            #     progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, self.generate_meta(), content_errors)

    def parse_annotation(self, annotation_path: str) -> list:
        frames = list()
        labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
        annotation_list = read_txt(annotation_path)
        num_iterations = len(annotation_list)
        current_img_id = self.global_idx + int(annotation_list[0].split(',')[0])
    #         content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
        for idx, line in enumerate(annotation_list):
            img_id, _, x_min, y_min, width, height, _ = line.split(',')
            img_id = self.global_idx + int(img_id)
            if current_img_id != img_id:
                frames.append(DetectionAnnotation(self.getImagePathById(current_img_id), np.array(
                    labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs)))
                labels.clear()
                x_mins.clear()
                y_mins.clear()
                x_maxs.clear()
                y_maxs.clear()
                current_img_id = img_id
            else:
                x_max = float(x_min) + float(width)
                y_max = float(y_min) + float(height)

                labels.append(int(1))
                x_mins.append(float(x_min))
                y_mins.append(float(y_min))
                x_maxs.append(float(x_max))
                y_maxs.append(float(y_max))

        frames.append(DetectionAnnotation(self.getImagePathById(current_img_id), np.array(
            labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs)))
        
        self.global_idx = current_img_id
        return frames

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

    def remove_invalid_datasets(self, annotation_files: list):
        annotation_files = list(filter(lambda x: re.search(r'MOT17-(.*?)-SDP', x).group(1) not in ['05', '06'], annotation_files))
        return annotation_files
