import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image
dataset_dir = osp.abspath("./data/")



import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage.transform import resize
ROOT_DIR = os.path.abspath("../../")
doc_dir = os.path.abspath("../../datasets/doc/")
input_dir = os.path.join(ROOT_DIR,"./data")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "pretrained_model_indiscapes.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join("/scratch/surendra/logs")
colors = [(0.0, 1.0, 0.00000000000000),(1.0, 0.0000000000000000, 0.0),(0.7999999999999998, 1.0, 0.0),(0.1999999999999993, 0.0, 1.0),(0.0, 0.40000000000000036, 1.0),(0.8000000000000007, 0.0, 1.0),(1.0, 0.0, 0.0),(1.0, 0.0, 0.5999999999999996),(0.20000000000000018, 1.0, 0.0),(0.0, 1.0, 1.0)]


class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids



class Gdataset(Dataset):

	def load_data(self, dataset_dir, subset):
		# Add classes. We have only one class to add.
		classes = ['Hole(Virtual)', 'Hole(Physical)', 'Character Line Segment', 'Physical Degradation',
		'Page Boundary', 'Character Component', 'Picture', 'Decorator', 'Library Marker']
		self.add_class("object", 1, "H-V")
		self.add_class("object", 2, "H")
		self.add_class("object", 3, "CLS")
		self.add_class("object", 4, "PD")
		self.add_class("object", 5, "PB")
		self.add_class("object", 6, "CC")
		self.add_class("object", 7, "P")
		self.add_class("object", 8, "D")
		self.add_class("object", 9, "LM")
		self.add_class("object", 10, "BL")
		# Train or validation dataset?
		# assert subset in ["train", "val", "test"]
		
		train = (subset=="train")
		val = (subset=="val")
		test = (subset=="test")
		i = 0
		if test:
			i = 439
		elif val:
			i = 352
		dataset_dir = os.path.join(dataset_dir, subset)
		# TODO:update the file name if required(to evaluate separately)
		annotations = json.load(
		open(os.path.join(doc_dir, subset, "via_region_data.json")))
		# annotations = json.load(open(os.path.join(dataset_dir, "via_region_data_bhoomi.json")))
		# annotations = json.load(open(os.path.join(dataset_dir, "via_region_data_PIH.json")))
		annotations = annotations["_via_img_metadata"]
		annotations = list(annotations.values())  # don't need the dict keys

		# The VIA tool saves images in the JSON even if they don't have any
		# annotations. Skip unannotated images.
		annotations = [a for a in annotations if a['regions']]
		IMAGE_PATH = []
		# Add images

		for a in annotations:
			class_ids = []
			# Get the x, y coordinates of points of the polygons that make up
			# the outline of each object instance. These are stores in the
			# shape_attributes (see json format above)
			# The if condition is needed to support VIA versions 1.x and 2.x.
			if type(a['regions']) is dict:
				polygons = [r['shape_attributes'] for r in a['regions'].values()]
				objects = [s['region_attributes'] for s in a['regions'].values()]
			else:
				polygons = [r['shape_attributes'] for r in a['regions']]
				objects = [s['region_attributes'] for s in a['regions']]

			# print(objects)
			classes = ['Hole(Virtual)', 'Hole(Physical)', 'Character Line Segment', 'Physical Degradation',
			'Page Boundary', 'Character Component', 'Picture', 'Decorator', 'Library Marker']
			for obj in objects:
				if(obj['Spatial Annotation'] == 'Hole(Virtual)'):
					class_ids.append(1)
				if(obj['Spatial Annotation'] == 'Hole(Physical)'):
					class_ids.append(2)
				if(obj['Spatial Annotation'] == 'Character Line Segment'):
					class_ids.append(3)
				if(obj['Spatial Annotation'] == 'Physical Degradation'):
					class_ids.append(4)
				if(obj['Spatial Annotation'] == 'Page Boundary'):
					class_ids.append(5)
				if(obj['Spatial Annotation'] == 'Character Component'):
					class_ids.append(6)
				if(obj['Spatial Annotation'] == 'Picture'):
					class_ids.append(7)

				if(obj['Spatial Annotation'] == 'Decorator'):
					class_ids.append(8)

				if(obj['Spatial Annotation'] == 'Library Marker'):
					class_ids.append(9)
				if(obj['Spatial Annotation'] == 'Boundary Line'):
					class_ids.append(10)

			# load_mask() needs the image size to convert polygons to masks.
			# Unfortunately, VIA doesn't include it in JSON, so we must read
			# the image. This is only managable since the dataset is tiny.
			ff = a['filename'].split('/')[-2:]
			# print(ff)
			flg = 0
			image_source = 1
			if ff[0] == 'illustrations':
				flg = 0
				# print('pih')
				# ff1 = '/PIH_images'+'/'+ff[1]
				image_path = input_dir + "/" + str(i) + ".npy"
				# print(image_path)
			else:
				image_path = input_dir + "/" + str(i) + ".npy"

			IMAGE_PATH.append(image_path)
			try:
				image = np.load(image_path)
			except Exception:
				print(image_path)
				print('Exception')
				i += 1
				continue

			height, width = [1024,1024]
		
			self.add_image(
			"object",
			image_id=a['filename'],  # use file name as a unique image id
			path=image_path,
			width=width, height=height,
			polygons=polygons,
			num_ids=class_ids,
			)
			i += 1

	def load_mask(self, image_id):
		"""Generate instance masks for an image.
	   Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		image_info = self.image_info[image_id]
		if image_info["source"] != "object":
			return super(self.__class__, self).load_mask(image_id)

		# Convert polygons to a bitmap mask of shape
		# [height, width, instance_count]
		info = self.image_info[image_id]
		num_ids = info['num_ids']
		mask = np.zeros( [info["height"], info["width"], len(info["polygons"])],
						dtype=np.uint8)
		for i, p in enumerate(info["polygons"]):
			# Get indexes of pixels inside the polygon and set them to 1
			try:
				rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
			except Exception:
				continue
			try:
				rr[rr>=info['height']] = info['height'] - 1
				cc[cc>=info['width']] = info['width'] - 1
				mask[rr, cc, i] = 1
			except Exception as e:
				print(e)

		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID only, we return an array of 1s
		num_ids = np.array(num_ids, dtype=np.int32)
		
		return mask.astype(np.bool), num_ids


class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        # self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # if not max_iters==None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = Gdataset()
        self.files.load_data(dataset_dir,se=1)
            # self.files.append({
            #     "img": img_file,
            #     "label": label_file,
            #     "name": name
            # })

    def __len__(self):
        return len(self.files.image_info)


    def __getitem__(self, index):
        datafiles = self.files.image_info[index]
    
        image = Image.open(datafiles['path']).convert('RGB')
        label = self.files.load_mask(index)
        name = datafiles['path']

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        # label = label.resize(self.crop_size, Image.NEAREST)
        label = cv2.resize(label,self.crop_size,interpolation = cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        # label_copy = 255 * np.ones(label.shape, dtype=np.float32)


        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), name