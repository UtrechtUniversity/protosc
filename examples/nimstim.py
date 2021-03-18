from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import re

from protosc.preprocessing import GreyScale, ViolaJones, CutCircle
from protosc.pipe_complex import PipeComplex
from protosc.feature_extraction import FourierFeatures
from protosc.io import ReadImage
grayscale = True


class NimStim:

<<<<<<< HEAD
    def __init__(self, stim_data_dir: Path):
        self.stim_data_dir = stim_data_dir
=======
    # Read all images
    files_path = list(self.stim_data_dir.glob('*'))
    files_path = [x for x in files_path if x.suffix.lower() == ".bmp"]

    files = list(x.stem for x in files_path)
>>>>>>> 2f81387adf972703601561e2c790b0bcec0742f1

    def create_csv(self, write=False):
        """ Create dataframe with all image IDs and sex, emotion, and mouth position depicted on image"""

        # Read all images
        files_path = list(self.stim_data_dir.glob('*.bmp'))
        files = list(x.stem for x in files_path)

        # Create codebook with the meaning of all abbreviations
        codebook = {0: {'female', 'male'},
                    1: {'calm', 'angry', 'happy', 'fear', 'sad', 'surprised', 'neutral', 'disgust'},
                    2: {'open', 'closed', 'exuberant'}}
        output = [[], [], []]

        # Decode sex, emotion, and mouth position depicted on image
        for file in files:
            file_parts = file.split('_')
            file_parts[0] = file_parts[0][-1]

            for i in range(0, 3):
                try:
                    code = re.findall('(?<=\')' + file_parts[i].lower() + '\w*', str(codebook[i]))[0]
                    output[i].append(code)
                except:
                    if file_parts[i].lower() == 'sp':
                        output[i].append('surprised')
                    elif file_parts[i].lower() == 'x':
                        output[i].append('exuberant')
                    else:
                        pass

        # Create dataframe with all collected info about the image
        df = {'file': files_path,
              'sex': output[0],
              'emotion': output[1],
              'mouth': output[2]}
        df = pd.DataFrame(df)
        df.index.name = 'picture_id'

        # Write dataframe to csv file
        if write:
            df.to_csv('overview.csv', index=True)

        return df

    def select_files(self, select:str, write):
        """ Put specified files through the pipeline"""

        overview = self.create_csv(write=write)
        variations = overview[select].unique()

        overview['class'] = ''
        for classification, variation in enumerate(variations):
            index = overview[overview[select] == variation].index
            overview.loc[index, 'class'] = classification

        x = np.array(overview['class'])
        files = np.array(overview['file'])

        return files, x

    def feature_matrix(self, output):
        """ Create Fourier Matrix: input = pipeline output, row = image, column = Feature """

        feature_matrix = pd.DataFrame()

        for image in range(len(output)):
            # If you only run 1 pipeline
            if isinstance(output[image], np.ndarray):
                data = pd.DataFrame(output[image]).T
                data['pipeline'] = 'pipe1'
                data['picture_id'] = image
                feature_matrix = feature_matrix.append(data)

            # If you run a pipe_complex
            else:
                for pipe in output[image].keys():
                    data = pd.DataFrame(output[image][pipe]).T
                    data['pipeline'] = pipe
                    data['picture_id'] = image
                    feature_matrix = feature_matrix.append(data)

        feature_matrix = feature_matrix.sort_values(['pipeline', 'picture_id']).set_index(['pipeline', 'picture_id'])

        return feature_matrix

    def execute(self, pipe_complex):
        """ Execute functions with corresponding pipeline"""

        # Put selected images through pipeline (e.g., classified on mouth (O, C, X))
        files, x = self.select_files(select='mouth', write=True)
        output = pipe_complex.execute(files)

        # Create feature matrix from pipeline output
        feature_matrix = self.feature_matrix(output)

        return feature_matrix, x, files


def main():
    parser = argparse.ArgumentParser(description='Validatie anonymization process.')
    parser.add_argument("--stim_data_dir", "-p", help="Enter path to Nimstim faces",
                        default=".")

    args = parser.parse_args()

    # Define pipeline
    pipe1 = ReadImage() * ViolaJones(20) * CutCircle() * FourierFeatures()
    pipe2 = ReadImage() * GreyScale() * ViolaJones(20) * CutCircle() * FourierFeatures()
    pipe_complex = pipe1 + pipe2

    nimstim = NimStim(Path(args.stim_data_dir))
    nimstim.execute(pipe_complex)


if __name__ == "__main__":
    main()
