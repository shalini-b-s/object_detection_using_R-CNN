Implementation of R-CNN Model for object detection from scratch.

In object detection, we have to classify the objects in an image and also locate where these objects are present in the image. 

There are two types of object detection networks: 

	* Two-stage network: R-CNN and its variants.
	* Single-stage network: YOLO. 

We have worked on two-stage network in this project.  

	> The initial stage of this network identifies region proposals: 
	
		* By using selective search segmentation algorithm, we will generate 2000 region proposals (bounding box) from an image that could contain an object. (x, y, width, height). 
		* IOU (Intersection over union) is calculated between each of these bounding boxes and ground truth bounding box (original object’s bounding box). 
		* Since we have decided to detect only one class (person) in an image, IOU > 0.7 is considered as an object (label = 1) and IOU < 0.3 as a background (label = 0). About 20 objects and 20 background are selected from a single image. 

	> The second stage classifies the above cropped images(objects and background) using transfer learning(Resnet model)
	

Dataset: COCO dataset 

	* Coco dataset is a large-scale image dataset containing 328,000 images of 80 object categories. 
	* The dataset was downloaded from fiftyone library. We have loaded all the images which contains person. These images may also contain other objects with person. 
	* Details regarding the bounding boxes of objects in an image, image_id, category_id, etc will be available in the annotation file. 


> region_proposal.ipynb --> Importing the dataset, creating a dataframe (image_id, bounding_box, image_path), dividing the dataset (training, validation, testing), creating regions using selective search segmentation, creating 20 object images with IOU > 0.7 and 20 background images with IOU < 0.3 from each image, and saving those cropped images. 

> cnn_model.ipynb --> loading the dataset, batching and balancing the dataset, image augmentation. Classifying objects and background by transfer learning (Resnet model). Achieved 90% accuracy and saved the best accuracy model as model1.pt. 

> inference.ipynb --> given an image, detect the objects present in that image. Using non-max suppression to select bounding box with maximum score for each object out of many overlapping objects. 

> utils.ipynb --> contains functions to calculate IOU and to save images to the given directory. 

