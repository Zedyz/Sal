# Saliency Prediction using GAN

This is a SalGAN implementation without the pre-trained VGG features written in PyTorch.

## Report

For all other information and a detailed report and metrics, see ```Final Report.pdf```

## Usage

Clone the repo. Download the trained generator weights [here](https://www.filemail.com/d/rtvlswuxeldsbtg). Run Test_Model.ipnyb using Jupyter Notebook and use functions ```saliency_from_link``` and ```saliency_from_val```

## Train it yourself
### Step 1:

Download the images and fixation maps from [SALICON](http://salicon.net/challenge-2017/) and save this data in a folder called ```data```.

### Step 2:
Clone this repository. Install dependencies.
```bash
pip install -r requirements.txt
```

### Step 3:

Rezise the images into 256x192

```bash
python resize_data.py
```

### Step 4:

Train the model

```bash
python main.py
```
