# protosc

Python library for automatic feature extraction. It is currently under active development. The overall pipeline looks as follows:

- preprocessing of the data
- feature extraction
- feature selection

Below we list from all these categories, what is currently developed or under development.

## IO

- Read image from file.

## Preprocessing

Currently only image processing preprocessing steps are included.

- convert to grey scale
- Apply Viola-Jones algorithm for face detection
- Cut a circle in the middle (face after VJ)


## Feature extraction

- Fourier transformation of image


## Feature selection

- Filter feature selection
- Combination of results from filter feature selection to final feature selection.

## Pipeline

The pipeline system allows for easy definition of pipelines:

```python
pipe1 = ReadImage()*GreyScale()*ViolaJones(add_perc=15)*FourierFeatures()
pipe2 = ReadImage()*ViolaJones()*FourierFeatures()
pipe_complex = pipe1 + pipe2
features = pipe_complex.execute(x)  # Here x can be a list of filenames
```