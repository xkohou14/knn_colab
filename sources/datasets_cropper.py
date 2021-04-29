import torch
from torch.utils import data

from cropper import Cropper
from couch import Couch, get_classes_in_image, directory, get_origin_name

import cv2
import os


class CropperSegmentationDataSet(data.Dataset):
    def __init__(self,
                 couch: Couch,
                 width: int,
                 height: int,
                 transform=None
                 ):
        self.couch = couch
        self.cache = CacheManager(width=width, height=height, couch_instance=couch)
        """
        This is a list of files (triplet: origin_image_path, classes_image_path, objects_image_path).
            Structure is: [
                (origin_image_path_1, classes_image_path_1, objects_image_path_1),
                (origin_image_path_2, classes_image_path_2, objects_image_path_2),
                .
                .
                .
            ] 
        For each element of this list is true that each element of this triplet is different (there is really class 
        template image, because dataset can contain any origins without class template and it is useless)
        """
        #files = couch.get_triplets_with_class_template(couch.train) + couch.get_triplets_with_class_template(couch.val)
        final_structure, files = self.cache.get_records_and_triplets()

        print("Initialization of files done: {} files".format(len(files)))

        """
        Contains structure of quartet: (origin_img, class_img, object_img, (x, y), transformation), where the last 
        element contains the coords of the crop in the cropper. Transformation is a value of rotation which should be 
        done with image before cropping.
        This is used for iterating over nonexistent cropped images
        """
        self.final_structure = final_structure

        self.cropper = Cropper(width=width, height=height)

        self.init_from_scratch(files)

        print("Initialization of final_structure done: {} records (images to train)".format(len(self.final_structure)))

        self.to_tensor=True
        self.transform=transform

    def init_from_scratch(self, files):
        """
        Run it when cache manager does not have a cache hit
        :files: files from Couch class - see its structure in init
                triplets: (origin_image_path, classes_image_path, objects_image_path)
        """
        for i, triplet in enumerate(files):
            if i % 5 == 0:
                print("Cropping iteration {} ({} %), final_structure_size: {}, {}".format(
                                                i, str((i/len(files))*100),
                                                len(self.final_structure),
                                                self.final_structure[-1] if len(self.final_structure) > 0 else "")
                )
            origin_img, class_img, object_img = triplet
            self.cropper.set_imgs(origin_img, class_img, object_img)
            while not self.cropper.is_finished:
                actual_coords = self.cropper.coords[0]
                _, class_img_cv2, _ = self.cropper.next_crop()
                if len(get_classes_in_image(class_img_cv2)) > 0:  # it has class so register it
                    self.final_structure.append((origin_img, class_img, object_img, actual_coords, 0))
                    self.cache.append_cache((origin_img, class_img, object_img, actual_coords, 0))

    def __len__(self):
        return len(self.final_structure)

    def get_images_from_final_structure(self, index, to_tensor=True):
        origin_img, class_img, object_img, coord, transformation = self.final_structure[index]  # get element from final_structure
        self.cropper.set_imgs(origin_img, class_img, object_img, transform=transformation)  # set the cropper image to analyze
        self.cropper.coords = [coord]  # override cropper coord by our one registered coord => only one image can be received
        return self.cropper.next_crop(convert_to_tensor=to_tensor)  # get this one crop

    def __getitem__(self,
                    index: int):
        origin_tensor, class_tensor, _ = self.get_images_from_final_structure(index, to_tensor=self.to_tensor)

        return origin_tensor, class_tensor


class CacheManager:

    def __init__(self, width: int, height: int, couch_instance: Couch, cache_dir: str = "dataset_cache"):
        self.couch = couch_instance
        self.width = width
        self.height = height
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cache_file_end_str = "{}x{}.csv".format(self.width, self.height)

        self.cache_file = None
        self.search_cache()

        """
        Cache dictionary stores key values
            key: origin_name
            value: final structure => (origin_img, class_img, object_img, (x, y), transformation)
        """
        self.cache_dictionary = {}
        self.init_cache()

    def init_cache(self):
        self.search_cache()

        def local_file(file):
            return os.path.join(self.couch.dataset, file)

        counter = 0
        if self.cache_file is not None:
            self.cache_dictionary = {}
            with open(self.cache_file) as opened_cache_file:
                for line in opened_cache_file:
                    origin_img_name, x, y, transformation = line.split(";", 3)
                    #origin_img, class_img, object_img = local_file(origin_img), local_file(class_img), local_file(object_img)
                    origin_img, class_img, object_img = self.couch.get_triplet(origin_img_name, self.couch.train + self.couch.val)
                    if origin_img is None:  # skip loading from cache if the Couch does not need it
                        continue

                    x, y, transformation = int(x), int(y), int(transformation)
                    if not get_origin_name(origin_img) in self.cache_dictionary.keys():
                        self.cache_dictionary[get_origin_name(origin_img)] = []
                    # add to dictionary
                    self.cache_dictionary[get_origin_name(origin_img)].append((origin_img, class_img, object_img, (x, y), transformation))
                    counter += 1
        print("Dictionary initialized (keys: {}: {}, size: {})".format(len(self.cache_dictionary), self.cache_dictionary.keys() if len(self.cache_dictionary) <= 5 else "", counter))

    def list_caches(self, dir_to_search=None):
        if dir_to_search is None:
            dir_to_search = self.cache_dir

        cache_files = []
        for filename in os.listdir(dir_to_search):
            if filename.endswith(self.cache_file_end_str):
                cache_files.append(os.path.join(dir_to_search, filename))
            else:
                try:
                    cache_files = cache_files + self.list_caches(os.path.join(dir_to_search, filename))
                except:
                    print("{} is not folder".format(os.path.join(dir_to_search, filename)))
        return cache_files

    def search_cache(self):
        caches = self.list_caches()
        if len(caches) > 0:
            print("Cache hit ({}x): {}".format(len(caches), caches[0]))
            self.cache_file = caches[0]
        else:
            print("Cache MISS")
            self.cache_file = None

    def append_cache(self, record):
        """
        Appends cache file by record
            record format is: (origin_img, class_img, object_img, (x, y), transformation)
        """
        origin_img, class_img, object_img, actual_coords, transformation = record
        x, y = actual_coords

        def file(path: str):
            # return os.path.basename(os.path.normpath(path))
            return path.split(self.couch.dataset)[1]  # return the path in dataset directory
        line = ";".join([get_origin_name(origin_img), str(x), str(y), str(transformation)])
        if self.cache_file is not None:
            with open(self.cache_file, 'a') as file:
                file.write("{}\n".format(line))
        else:
            self.cache_file = os.path.join(self.cache_dir, "cache_" + self.cache_file_end_str)
            print("Cache manager Creates new cache file {}".format(self.cache_file))
            self.append_cache(record)

    def get_records_and_triplets(self):
        """
        Returns a tuple: satisfied_records_array, unsatisfied_triplets_array
            satisfied_record format is: (origin_img, class_img, object_img, (x, y), transformation)
            unsatisfied_triplet format is: (origin_img, class_img, object_img)
        """
        satisfied = []
        unsatisfied = []

        all_files = self.couch.train + self.couch.val
        for _, origin_name in enumerate(self.couch.get_iterable_origins(all_files)):
            if origin_name in self.cache_dictionary.keys():
                for _, record in enumerate(self.cache_dictionary[origin_name]):
                    satisfied.append(record)
            elif len(origin_name) > 0:
                origin_img, class_img, object_img = self.couch.get_triplet(origin_name, all_files)
                unsatisfied.append((origin_img, class_img, object_img))
                print("Unsatisfied: {} ({})".format(origin_name, origin_img))
        print("get_records_and_triplets: satisfied: {} records found, unsatisfied: {} files".format(len(satisfied), len(unsatisfied)))
        return satisfied, unsatisfied


if __name__ == '__main__':
    couch = Couch(directory)
    couch1, couch2 = couch.split_couch(0.008)
    dataset = CropperSegmentationDataSet(couch1, width=512, height=512)
    dataset.to_tensor = False
    origin, classes = dataset[1]
    cv2.imshow("Origin", origin)
    cv2.imshow("Classes", classes)
    cv2.waitKey(0)
