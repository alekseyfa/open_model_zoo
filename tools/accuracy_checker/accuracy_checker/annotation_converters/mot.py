import numpy as np
import os
from .format_converter import BaseFormatConverter, ConverterReturn
from ..utils import read_txt, check_file_existence
from ..config import PathField, BoolField
from ..representation import DetectionAnnotation

# kek = 0

class MOTConverter(BaseFormatConverter):
    __provider__ = 'mot'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'images_dir': PathField(optional=True, is_directory=True, description='Images directory'),
            'annotation_file': PathField(optional=False, is_directory=True, description='Annotation file'),
            'has_background': BoolField(optional=True, default=False, description='Indicator of background'),
            'add_background_to_label_id': BoolField(
                optional=True, default=False, description='Indicator that need shift labels'
            )
        })
        return params

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.has_background = self.get_value_from_config('has_background')
        self.shift_labels = self.get_value_from_config('add_background_to_label_id')
        self.kek = 0

    def getImagePathById(self, image_id, prefix="", sufix=".jpg"):
            # "000000000257"
            im_id_str = str(image_id)
            zeros_str = '0' * (6 - len(im_id_str))
            im_id_str = zeros_str + im_id_str
            image_path = prefix + im_id_str + sufix
            return image_path

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = None if not check_content else []
        annotations = list()
        labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
        annotation_list = read_txt(os.path.join(self.annotation_file, 'det.txt'))
        num_iterations = len(annotation_list)
        current_img_id = annotation_list[0].split(',')[0]
    #         content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
        for idx, line in enumerate(annotation_list):
            # if kek < 10:
            #     print(line)
            img_id, _, x_min, y_min, width, height, _ = line.split(',')
            if current_img_id != img_id:
                if self.kek < 10:
                    print(self.getImagePathById(current_img_id), np.array(labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs))
                    self.kek += 1
                annotations.append(DetectionAnnotation(self.getImagePathById(current_img_id), np.array(labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs)))
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

            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        annotations.append(DetectionAnnotation(self.getImagePathById(current_img_id), np.array(labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs)))
        return ConverterReturn(annotations, self.generate_meta(), content_errors)

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
