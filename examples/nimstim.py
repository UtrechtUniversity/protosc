from pathlib import Path
import numpy as np
import re
import csv

from protosc.preprocessing import GreyScale, ViolaJones, CutCircle
from protosc.feature_extraction import FourierFeatures
from protosc.io import ReadImage


def create_csv(stim_data_dir, write=False):
    """ Create dataframe with all image IDs and sex, emotion, and mouth position depicted on image"""

    # Read all images
    files_path = list(stim_data_dir.glob('*'))
    files_path = [x for x in files_path if x.suffix.lower() == ".bmp"]

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
                code = re.findall(
                    r'(?<=\')' + file_parts[i].lower() + r'\w*', str(codebook[i]))[0]
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

    # Write dataframe to csv file
    if write:
        df = []
        for i in range(len(files_path)):
            data = {'file': files_path[i],
                    'sex': output[0][i],
                    'emotion': output[1][i],
                    'mouth': output[2][i]}
            df.append(data)

        csv_columns = ['file', 'sex', 'emotion', 'mouth']
        csv_file = "overview.csv"

        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(df)
        except IOError:
            print('error')
    return df


def select_files(stim_data_dir, select: str, write=False):
    """ Put specified files through the pipeline"""

    overview = create_csv(stim_data_dir, write=write)
    variations = overview[select].unique()

    overview['class'] = ''
    for classification, variation in enumerate(variations):
        index = overview[overview[select] == variation].index
        overview.loc[index, 'class'] = classification

    y = np.array(overview['class'])
    files = np.array(overview['file'])

    return files, y


def feature_matrix(output, pipe_complex):
    """ Create Feature Matrix: input = pipeline output, output = list (per pipeline) with feature_arrays per image """

    final = []

    # If you run only one pipeline
    if not isinstance(output[0], dict):
        final.append(output)

    # If you run multiple pipelines
    else:
        for pipe in output[0].keys():
            data = np.array([output[image][pipe]
                             for image in range(len(output))])
            final.append(data)

    return final


def execute(pipe_complex, stim_data_dir: Path, select: str, write=False):
    """ Execute functions with corresponding pipeline"""

    # Put selected images through pipeline (e.g., classified on mouth (O, C, X))
    print('Filtering images...')
    files, y = select_files(stim_data_dir, select=select, write=write)

    print('Putting images through pipeline...')
    output = pipe_complex.execute(files)

    # Create feature matrix from pipeline output
    print('Creating feature matrix...')
    feature_array = feature_matrix(output, pipe_complex)

    return feature_array, y, files


def main():

    # Define pipeline
    pipe1 = ReadImage() * ViolaJones(20) * CutCircle() * FourierFeatures()
    pipe2 = ReadImage() * GreyScale() * ViolaJones(20) * \
        CutCircle() * FourierFeatures()
    pipe_complex = pipe1 + pipe2

    # Define path to images
    stim_data_dir = Path("..", "data", "Nimstim faces")

    # Execute pipeline
    execute(pipe_complex, stim_data_dir, select='mouth', write=False)


if __name__ == "__main__":
    main()
