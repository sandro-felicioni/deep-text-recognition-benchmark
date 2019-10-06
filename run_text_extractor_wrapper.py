import os
import time

from PIL import Image

from text_extractor_deep_text_recognition.text_extractor_wrapper import TextExtractorWrapper

def list_files(path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
        break
    return img_files

if __name__ == '__main__':

    # Initialize model
    arguments = ["--Transformation=TPS", "--FeatureExtraction=ResNet", "--SequenceModeling=BiLSTM", "--Prediction=Attn", "--image_folder=demo_image/", "--saved_model=pretrained_models/TPS-ResNet-BiLSTM-Attn.pth"]
    TextExtractorWrapper = TextExtractorWrapper(arguments)

    # Run against test data
    t = time.time()
    image_list = list_files("./demo_image")
    for k, image_path in enumerate(image_list):
        print("Test image %d/%d: %s" % (k+1, len(image_list), image_path))
        image = Image.open(image_path)
        text = TextExtractorWrapper.predict(image)
        print("\t\t\tResult: %s" % text)

    print("elapsed time : {}s".format(time.time() - t))