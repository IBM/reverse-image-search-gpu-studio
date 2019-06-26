# Accelerate Reverse Image Search with GPU for feature extraction

In this code pattern, we will guide you through the process of analyzing an image dataset using a pre-trained convolution network (VGG16) and extracting feature vectors for each image using a Jupyter notebook.
This is a computationally expensive process, which takes 300 times longer on a CPU versus a GPU. We'll use the GPU environment on Watson Studio or on your local machine to accelerate feature extraction.
Post analysis we try to demonstrate 'reverse image search', one of the widely popular applications of image analysis. Reverse image search is a content-based image retrieval (CBIR) query technique that involves providing the CBIR system with a sample image that it will then base its search upon; in terms of information retrieval, the sample image is what formulates a search query. In particular, reverse image search is characterized by a lack of search terms.[1]

[1] https://en.wikipedia.org/wiki/Reverse_image_search

When you have completed this code pattern, you will understand how to:

* Use GPU acceleration in Watson Studio or locally to greatly improve performance of feature extraction.
* Download [VGG16 pre-trained model](https://keras.io/applications/#vgg16) using keras.
* Perform Feature Extraction. Here we remove the last layer ie.,the softmax classification layer so our output model now has only 12 layers and the last layer would be fc2(Dense) a fully connected layer.
* Get feature vectors for all the images then scale them down using [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
* Use cosine distance between pca features to compare the query image to 5 number of closest images and return them as thumbnails.

## Watch the Video:

[![video](http://img.youtube.com/vi/FpWsLvXFCy0/0.jpg)](https://youtu.be/FpWsLvXFCy0)

## Steps

1. [Clone the repository](#1-clone-the-repository)
2. Perform either 2a or 2b:  
  2a. [Create a notebook in IBM Watson Studio](#2a-create-a-notebook-in-ibm-watson-studio)  
  2b. [Create notebook locally](#2b-create-notebook-locally)  
3. [Run the notebook](#3-run-the-notebook)

### 1. Clone the repository

```bash
git clone https://github.com/IBM/reverse-image-search-gpu-studio
cd reverse-image-search-gpu-studio
```

### 2. Create a notebook

Either in Watson Studio or locally.

#### 2a. Create a notebook in IBM Watson Studio

* Sign up for IBM's [Watson Studio](https://dataplatform.cloud.ibm.com/). By creating a project in Watson Studio a free tier ``Object Storage`` service will be created in your IBM Cloud account. Take note of your service names as you will need to select them in the following steps.

> Note: When creating your Object Storage service, select the ``Free`` storage type in order to avoid having to pay an upgrade fee.

* Create a new Project in Watson Studio (New --> Standard project)

![Creating a project](doc/source/images/createProject.gif)

* Create a GPU Environment (Environment --> New Environment --> GPU Beta)

![Creating GPU Environment](doc/source/images/createEnv.gif)

* Create a new Notebook (Add to project --> Notebook --> from url)
* Provision the notebook on newly created GPU Environment

![GPU Notebook](doc/source/images/createNotebook.gif)

* Stop the Environment Post usage

![Stop environment](doc/source/images/stopEnv.gif)


#### 2b. Create notebook Locally

* Clone the repository

```bash
git clone https://github.com/IBM/reverse-image-search-gpu-studio
```

* Navigate into the directory

```bash
cd reverse-image-search-gpu-studio
```

* Run using Jupyter notebooks, choosing the `data/ReverseImageSearch.ipynb` notebook

```bash
jupyter notebook
```

### 3. Run the notebook

When a notebook is executed, what is actually happening is that each code cell in
the notebook is executed, in order, from top to bottom.

Each code cell is selectable and is preceded by a tag in the left margin. The tag
format is `In [x]:`. Depending on the state of the notebook, the `x` can be:

* A blank, this indicates that the cell has never been executed.
* A number, this number represents the relative order this code step was executed.
* A `*`, this indicates that the cell is currently executing.

There are several ways to execute the code cells in your notebook:

* One cell at a time.
  * Select the cell, and then press the `Play` button in the toolbar.
* Batch mode, in sequential order.
  * From the `Cell` menu bar, there are several options available. For example, you
    can `Run All` cells in your notebook, or you can `Run All Below`, that will
    start executing from the first cell under the currently selected cell, and then
    continue executing all cells that follow.
* At a scheduled time.
  * Press the `Schedule` button located in the top right section of your notebook
    panel. Here you can schedule your notebook to be executed once at some future
    time, or repeatedly at your specified interval.

## Sample Output:

With GPU on Watson Studio:

```
tic = time.time()
features = []
for i, image_path in enumerate(images):
  if i%500 == 0:
    toc = time.time()
    elap = toc-tic;
    print("analyzing image %d / %d. Time taken : %4.4f seconds"%(i,len(images),elap))
    tic= time.time()
  img,x = load_image(image_path)
  feat = feat_extractor.predict(x)[0]
  features.append(feat)
print('finished extracting features for %d images' % len(images))
```
```
analyzing image 0 / 9144. Time taken : 0.0001 seconds
analyzing image 500 / 9144. Time taken : 3.8999 seconds
analyzing image 1000 / 9144. Time taken : 3.8443 seconds
analyzing image 1500 / 9144. Time taken : 3.9378 seconds
analyzing image 2000 / 9144. Time taken : 4.0111 seconds
analyzing image 2500 / 9144. Time taken : 3.8682 seconds
analyzing image 3000 / 9144. Time taken : 4.7688 seconds
analyzing image 3500 / 9144. Time taken : 3.9087 seconds
analyzing image 4000 / 9144. Time taken : 3.8472 seconds
analyzing image 4500 / 9144. Time taken : 3.9209 seconds
analyzing image 5000 / 9144. Time taken : 3.6958 seconds
analyzing image 5500 / 9144. Time taken : 3.8112 seconds
analyzing image 6000 / 9144. Time taken : 3.9800 seconds
analyzing image 6500 / 9144. Time taken : 3.9112 seconds
analyzing image 7000 / 9144. Time taken : 3.9345 seconds
analyzing image 7500 / 9144. Time taken : 3.9807 seconds
analyzing image 8000 / 9144. Time taken : 4.5466 seconds
analyzing image 8500 / 9144. Time taken : 3.9019 seconds
analyzing image 9000 / 9144. Time taken : 3.9125 seconds
finished extracting features for 9144 images
```

Without GPU (CPU only):
```
analyzing image 0 / 9144. Time taken : 0.0000 seconds
analyzing image 500 / 9144. Time taken : 1446.1000 seconds
analyzing image 1000 / 9144. Time taken : 1467.4600 seconds
analyzing image 1500 / 9144. Time taken : 1441.9900 seconds
analyzing image 2000 / 9144. Time taken : 1415.6000 seconds
analyzing image 2500 / 9144. Time taken : 1435.7600 seconds
analyzing image 3000 / 9144. Time taken : 1405.6100 seconds
analyzing image 3500 / 9144. Time taken : 1453.4300 seconds
analyzing image 4000 / 9144. Time taken : 1385.6300 seconds
analyzing image 4500 / 9144. Time taken : 1398.6800 seconds
analyzing image 5000 / 9144. Time taken : 1409.2400 seconds
analyzing image 5500 / 9144. Time taken : 1409.0400 seconds
analyzing image 6000 / 9144. Time taken : 1470.3800 seconds
analyzing image 6500 / 9144. Time taken : 1436.4900 seconds
analyzing image 7000 / 9144. Time taken : 1465.0100 seconds
analyzing image 7500 / 9144. Time taken : 1433.5900 seconds
analyzing image 8000 / 9144. Time taken : 1454.0300 seconds
analyzing image 8500 / 9144. Time taken : 1425.7300 seconds
analyzing image 9000 / 9144. Time taken : 1446.3600 seconds
finished extracting features for 9144 images
```

[Example Notebook](examples/ReverseImageSearchExample.ipynb)

## License

This code pattern is licensed under the Apache License, Version 2. Separate third-party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1](https://developercertificate.org/) and the [Apache License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache License FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)
