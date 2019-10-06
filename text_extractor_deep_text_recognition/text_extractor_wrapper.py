import string
import argparse
from typing import List

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from text_extractor_deep_text_recognition.utils import CTCLabelConverter, AttnLabelConverter
from text_extractor_deep_text_recognition.dataset import RawDataset, AlignCollate, RawImageDataset
from text_extractor_deep_text_recognition.model import Model

class TextExtractorWrapper:

    def __init__(self, arguments: List[str]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
        parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

        self.opt = parser.parse_args(arguments)

        """ vocab / character number configuration """
        if self.opt.sensitive:
            self.opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()

        """ model configuration """
        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        self.model = Model(self.opt)
        print('model input parameters', self.opt.imgH, self.opt.imgW, self.opt.num_fiducial, self.opt.input_channel, self.opt.output_channel,
              self.opt.hidden_size, self.opt.num_class, self.opt.batch_max_length, self.opt.Transformation, self.opt.FeatureExtraction,
              self.opt.SequenceModeling, self.opt.Prediction)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        print('loading pretrained model from %s' % self.opt.saved_model)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.opt.saved_model))
        else:
            self.model.load_state_dict(torch.load(self.opt.saved_model, map_location="cpu"))

    def predict(self, image):
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
        demo_data = RawImageDataset(image=image, opt=self.opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=AlignCollate_demo, pin_memory=True)

        # predict
        self.model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

                if 'CTC' in self.opt.Prediction:
                    preds = self.model(image, text_for_pred).log_softmax(2)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.permute(1, 0, 2).max(2)
                    preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                    preds_str = self.converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)

                results = []
                for img_name, pred in zip(image_path_list, preds_str):
                    if 'Attn' in self.opt.Prediction:
                        pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

                    results.append(pred)

                assert len(results) == 1
                return results[0]  # should anyway never have more than 1 results if the input is a single image
