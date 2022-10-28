Object detection with R-CNN Model.

Framework: PyTorch, Google Colab

In object detection, we have to classify the objects in an image and also locate where these objects are present in the image. 

There are two types of object detection networks: 

	* Two-stage network: R-CNN and its variants.
	* Single-stage network: YOLO. 

I have worked on two-stage network in this project.  

	> The initial stage of this network identifies region proposals: 
	
		* By using selective search segmentation algorithm, we can generate number of regions (bounding box) of an image that could contain an object. (x, y, width, height). 
		* IOU (Intersection over union) is calculated between each of these bounding boxes and ground truth bounding box (original image’s bounding box). 
		* Since I have decided to detect only person in an image, IOU > 0.7 is considered as an object and IOU < 0.3 as a background. 

	> The second stage classifies the objects within region proposals: 
	
		* The object and the background are classified using transfer learning (Resnet model).

Dataset: COCO dataset 

	* Coco dataset is a large-scale image dataset containing 328,000 images of everyday objects and humans. 
	* I have imported the dataset from fiftyone library with only images containing person.
	* Annotation file contains information on the different categories, image id, bounding box for each object.  

> region_proposal.ipynb --> Importing the dataset, creating regions using selective search segmentation, creating 20 object images with IOU > 0.7 and 20 background images with IOU < 0.3 from each image, and saving those objects and background. 

> cnn_model.ipynb --> loading the dataset, batching and balancing the dataset, image augmentation. Classifying objects and background by transfer learning (Resnet model). Achieved 90% accuracy and savied the best accuracy model as model1.pt. 

> inference.ipynb --> given an image, detect the objects present in that image. Using non-max suppression to select bounding box with maximum score for each object out of many overlapping objects. 

> utils.ipynb --> contains functions to calculate IOU and to save images to the given directory. 

