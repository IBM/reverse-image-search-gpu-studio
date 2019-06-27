# Accelerate Reverse Image Search with GPU for feature extraction

Extract feature vectors for machine learning using GPU for increased performance.

# Use the VGG16 pre-trained convolutional network to extract feature vectors for images, allowing reverse image search and demonstrating the performance acceleration that comes with using GPU for proccessing.

# by Krishna Balaga and Scott D'Angelo

* krbalaga@in.ibm.com
* scott.dangelo@ibm.com

# URLs

### https://github.com/IBM/reverse-image-search-gpu-studio

> "Get the code": https://github.com/IBM/reverse-image-search-gpu-studio

# Summary

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

# Technologies

* Artificial Intelligence. Create apps that accelerate, enhance, and scale the human expertise.
* Deep Learning. Create, train, and deploy self-learning models.
* Machine Learning. Teach systems to learn without them being explicitly programmed.

# Description

Machine learning algorithms provide us many useful tools that solve real world problems. One of the domains that ML has had great success with is image recognition. By using computational power to identify images, and compare to other images, we can perform importatn tasks that a few years ago could only be done by humans.
Engineers and data scientists who work with image recognition encounter a few challenges that provide limits to the work that can be done with machine learning algorithms. The biggest limitation is the time and computational power that are required to create the machine learning layers in a deep neural network, which are required for image recognition. Whereas large data sets and complex algorithms can be run on many common hardware configurations, the time required for creating neural layers, training a model, and feature extraction can be prohibitively high.
The use of the Graphics Processesing Unit to perform many computations in parallel has initially revolutionized the world of computer graphics, but the discovery that the same GPUs can also be used to accelerate the performance of machine learning tasks has had a similar effect on the world of Artificial Intelligence. In this code pattern, we demonstrate a common task for Content Based Image Retrieval (CIBR), and compare running only on the CPU versus the increased perfomance obtained when running on a GPU.

# Flow

> Upload a draft architecture diagram to this issue. Remember to include numbers in the diagram to represent the flow steps that you provide below the diagram. A graphic designer will use your draft to create the production-ready image.

1. Flow step 1
2. Flow step 2
3. Flow step 3

# Instructions

> Find the detailed steps for this pattern in the [README.md](https://github.com/IBM/reverse-image-search-gpu-studio/blob/master/README.md). The steps will show you how to:

1. Clone the repository
2. Perform either 2a or 2b:
  2a. Create a notebook in IBM Watson Studio
  2b. Create notebook locally
3. Run the notebook

# Components and services

* [Jupyter Notebook](https://jupyter.org/) An open-source web application that suppots interactive data science and scientific computing across all programming languages.
* [Keras](https://keras.io) The Python Deep Learning library.
* [Watson Studio](https://dataplatform.cloud.ibm.com) IBM's integrated hybrid environment that provides flexible data science tools to build and train AI models and prepare and analyze data.

# Runtimes

* Python3
* GPU

# Related IBM Developer content

* [GPU programming made easy with OpenMP on IBM POWER](https://developer.ibm.com/articles/gpu-programming-with-openmp/) Architecture-independent programming using OpenMP.
* [Set up a GPU enabled Kubernetes cluster with Kubeadm](https://developer.ibm.com/tutorials/k8s-kubeadm-gpu-setup) Learn how to set up GPUs in a Kubernetes cluster in IBM Cloud.

# Announcement
> Every pattern must have an announcement post that introduces it. The announcement should explain why the pattern is important or useful. The announcement is an invitation to try the pattern; you can expand on why you created the pattern, discuss any challenges that you overcame, or expand on the technologies that you're using.

> *Announcements should be at least 2-3 paragraphs*
