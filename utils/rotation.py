import torch
import torch.utils.data
from torchvision import datasets
import numpy as np

# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
	return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
	if label == 1:
		img = tensor_rot_90(img)
	elif label == 2:
		img = tensor_rot_180(img)
	elif label == 3:
		img = tensor_rot_270(img)
	return img

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def rotate_batch(batch, label='rand'):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels

class RotateImageFolder(datasets.ImageFolder):
	def __init__(self, traindir, train_transform, original=True, rotation=True, rotation_transform=None):
		super(RotateImageFolder, self).__init__(traindir, train_transform)
		self.original = original
		self.rotation = rotation
		self.rotation_transform = rotation_transform		

	def __getitem__(self, index):
		path, target = self.imgs[index]
		img_input = self.loader(path)

		if self.transform is not None:
			img = self.transform(img_input)
		else:
			img = img_input

		results = []
		if self.original:
			results.append(img)
			results.append(target)
		if self.rotation:
			if self.rotation_transform is not None:
				img = self.rotation_transform(img_input)
			target_ssh = np.random.randint(0, 4, 1)[0]
			img_ssh = rotate_single_with_label(img, target_ssh)
			results.append(img_ssh)
			results.append(target_ssh)
		return results

	def switch_mode(self, original, rotation):
		self.original = original
		self.rotation = rotation

class RotateImageFolder_csv(torch.utils.data):

	def __init__(self, csv_path, traindir, train_transform, original=True, rotation=True, rotation_transform=None):
		super(RotateImageFolder, self).__init__(traindir, train_transform)
		self.original = original
		self.rotation = rotation
		self.rotation_transform = rotation_transform		

        self.filenames = []
        self.classes = []
        self.labels = []
        self.target = []
        with open(csv_path,mode="r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row["filename"] != "filename" and row["label"] != "label":
                    self.filenames.append(row["filename"])
                    self.labels.append(row["label"])
                    if row["label"] not in self.classes:
                        self.classes.append(row["label"])
        self.classes.sort()
        for label in self.labels:
            self.target.append(self.classes.index(label))
        self.data_dir = traindir
        self.transform = transform

    def get_image_from_folder(self, name):
        image = Image.open(os.path.join(self.data_dir, name)).convert("RGB")
        return image

    def __len__(self):
        return len(self.filenames)

	def get_item(self, index):
        Y = self.labels[index]
        target = self.target[index]
        X = self.get_image_from_folder(os.path.join(Y,self.filenames[index]))
        if self.transform is not None:
            X = self.transform(X)
        return X,target

	def __getitem__(self, index):
		path, target = self.get_item(index)
		img_input = self.loader(path)

		if self.transform is not None:
			img = self.transform(img_input)
		else:
			img = img_input

		results = []
		if self.original:
			results.append(img)
			results.append(target)
		if self.rotation:
			if self.rotation_transform is not None:
				img = self.rotation_transform(img_input)
			target_ssh = np.random.randint(0, 4, 1)[0]
			img_ssh = rotate_single_with_label(img, target_ssh)
			results.append(img_ssh)
			results.append(target_ssh)
		return results

	def switch_mode(self, original, rotation):
		self.original = original
		self.rotation = rotation