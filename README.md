# Opti-Car

Car price predictor using a convolutional neural network. This was for the final assignment for the Neural Networks course.

By s3301419, s3814254, s3593673

## Dataset

The car pictures used for this project are from Nicolas Gervais' [Car Connection Picture Dataset](https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper). This dataset originally consists of roughly 60,000 images which are all tagged with the suggested retail price ([MSRP](https://en.wikipedia.org/wiki/List_price)) of the cars.

## Requirements

| **library**                                              | **version used** | **scripts**          |
|:--------------------------------------------------------:|:----------------:|:--------------------:|
| python                                                   | 3.7.3            | *                    |
| [tensorflow](https://www.tensorflow.org/)                | 2.2.0            | opticar.py           |
| [numpy](https://numpy.org/)                              | 1.16.2           | opticar.py           |
| [opencv-python](https://opencv.org/)                     | 4.1.0            | opticar.py           |
| [matplotlib](https://matplotlib.org/)                    | 3.0.3            | opticar.py           |
| [tqdm](https://github.com/tqdm/tqdm)                     | 4.45.0           | opticar.py           |
| [Pillow](https://pillow.readthedocs.io/en/stable/)       | 5.4.1            | sortbg.py            |
| [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/) | 0.9.50           | auto.py, autoinfo.py |
| [pynput](https://github.com/moses-palmer/pynput)         | 1.6.8            | autoinfo.py          |

To install all of the above:

```bash
$ pip install tensorflow numpy opencv-python matplotlib tqdm pillow pyautogui pynput
```

## How to run

To train the model, download the dataset, and pass the dataset-directory and number of epochs to train to the script like so:

```bash
# example: $ python3 opticar.py train "cars/images/" 30
$ python3 opticar.py train DATASET-DIR NUM-EPOCHS
```

To use a trained model to predict prices, place the model's weight file `weights.h5` into the same directory as the script, and place all of the car images that you want to run the model on in another directory. Then run the script like so:

```bash
# example: $ python3 opticar.py "cars/images/"
$ python3 opticar.py IMAGES-DIR
```
