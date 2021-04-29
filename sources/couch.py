import os
import json, cv2, re, imutils
from skimage.metrics import structural_similarity
from cropper import Cropper
from PIL import Image
import numpy as np
from random import shuffle

directory = "../../data/"


def has_origin(name, origin_name):
    """
    Checks if the name contains the specific string
    :param name:
    :param origin_name:
    :return: True or False
    """
    return origin_name in name


def get_origin_name(name):
    """
    Returns the name of the picture without path
    :param name: name should contain P[0-9]{4}.png
    :return:
    """
    str_f = re.search(r"P[0-9]{4}.png", name)
    if str_f is not None:
        str_f = str_f.group(0)
    if str_f is None:
        return ""
    else:
        return re.search(r"P[0-9]{4}", str_f).group(0)


def get_imgs_difference_structural_similarity(reference, to_compare):
    """
    Returns the score of the structural similarity
    from: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    :param reference: reference image
    :param to_compare: generated image by network
    :return: score (<-1;1> and 1 is the perfect match), reference image, compared image, diff highlight, thresh
    """
    img1 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(to_compare, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return score, img1, img2, diff, thresh


def get_imgs_difference_iou(reference, to_compare):
    """
    Returns the score of the intersection-over-union => union_area / (covered_area_in_both_pictures), if they are
    the same union_area = covered_area_in_both_pictures and result is 1 (perfect match)
    from: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    :param reference: reference image
    :param to_compare: generated image by network
    :return: score (<0;1> and 1 is the perfect match)
    """
    ref_f = "reference_tmp_img.png"
    com_f = "to_compare_tmp_img.png"
    cv2.imwrite(ref_f, reference)
    cv2.imwrite(com_f, to_compare)

    ref_mask = np.array(Image.open(ref_f)).reshape(-1, 3)
    com_mask = np.array(Image.open(com_f)).reshape(-1, 3)

    ref_c = 0
    com_c = 0
    both_c = 0

    ref_cls = get_classes_in_image(reference)
    com_cls = get_classes_in_image(to_compare)

    def not_background(triplet) -> bool:
        r, g, b = triplet
        return r != 0 or g != 0 or b != 0

    for i, ref_triplet in enumerate(ref_mask):
        com_triplet = com_mask[i]
        if not_background(ref_triplet) and not_background(com_triplet):
            both_c += 1
        else:
            if not_background(ref_triplet):
                ref_c += 1
            if not_background(com_triplet):
                com_c += 1

    os.remove(ref_f)
    os.remove(com_f)

    res_iou = both_c / (com_c + ref_c + both_c)
    print("both: {}, ref: {}, com: {}, res: {}".format(both_c, ref_c, com_c, res_iou))

    return res_iou


def get_classes_in_image(image):
    """
    Gets cv2 image and returns list of RGBs
    :param image: cv2 image
    :return: list of RGBs for each class
    """
    # img_f = "img_tmp_img.png"
    # cv2.imwrite(img_f, image)
    # mask = Image.open(img_f)
    mask = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = np.array(mask)
    #print(mask.reshape(-1, 3)[:5])
    mask = mask.reshape(-1,3)
    # get classes which are at img
    cls_ids = np.unique(mask, axis=0)

    # remove background
    cls_ids = cls_ids[1:]

    #os.remove(img_f)
    return cls_ids


def split_data(data, ratio, shuffle_flag):
    """
    Splits dataset into 2 chunks
    It returns at least 1 record for each result
    """
    if len(data) <= 1:
        return data[:], data[:]

    first_len = int(round(ratio * len(data)))
    copy_data = data[:]
    if shuffle_flag:
        shuffle(copy_data)

    # copy_data1 = [copy_data[0]] if len(copy_data[:first_len]) <= 1 else copy_data[:first_len]
    # copy_data2 = [copy_data[-1]] if len(copy_data[first_len:]) <= 1 else copy_data[first_len:]
    if first_len == 0:  # every dataset has at least 1 record but they don't have union
        first_len += 1
    elif first_len == len(copy_data):
        first_len -= 1
    return copy_data[:first_len], copy_data[first_len:]


class Couch:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train = []
        self.val = []
        self.test = []
        self.init_subsets()

    def init_subsets(self):
        """
        Reset sets of datas (train, val, test)
        :return:
        """
        self.train, self.val, self.test = [], [], []
        data_to_iterate = self.list_dir()
        data_to_iterate.sort()
        for index, dato in enumerate(data_to_iterate):
            if "test" in dato:
                self.test.append(dato)
            elif "val" in dato:
                self.val.append(dato)
            elif "train" in dato:
                self.train.append(dato)

        self.train.sort()
        self.test.sort()
        self.val.sort()

    def list_dir(self, dir_to_search=None):
        if dir_to_search is None:
            dir_to_search = self.dataset

        local_dataset = []
        for filename in os.listdir(dir_to_search):
            if filename.endswith(".png"):
                local_dataset.append(os.path.join(dir_to_search, filename))
            else:
                try:
                    local_dataset = local_dataset + self.list_dir(dir_to_search + filename + "/")
                except:
                    try:
                        local_dataset = local_dataset + self.list_dir(os.path.join(dir_to_search, filename))
                    except:
                        print("{} is not folder".format(dir_to_search + filename + "/"))
        return local_dataset

    def get_iterable_evals(self):
        """
        Returns pure list of file names for which we can iterate and get the files by get_val_triple
        :return:
        """
        set_to_ret = set({})
        for i, val in enumerate(self.val):
            set_to_ret.add(get_origin_name(val))

        return list(set_to_ret)

    def get_val_triple(self, origin_name):
        """
        Returns the triplet of images for origin_name
        :param origin_name:
        :return: triplet where the pictures are: origin_image_path, classes_image_path (each class has its own color), objects_image_path (each object has its own color)
        """
        for i, val in enumerate(self.val):
            if has_origin(val, origin_name):
                filtered_files = list(filter(lambda name: has_origin(name, origin_name), self.val))

                if len(filtered_files) >= 3:
                    return filtered_files[0], filtered_files[1], filtered_files[2]
                elif len(filtered_files) >= 1:
                    return filtered_files[0], filtered_files[0], filtered_files[0]

    def get_iterable_trains(self):
        """
        Returns pure list of file names for which we can iterate and get the files by get_train_triple
        :return:
        """
        set_to_ret = set({})
        for i, val in enumerate(self.train):
            set_to_ret.add(get_origin_name(val))

        return list(set_to_ret)

    def get_train_triple(self, origin_name):
        """
        Returns the triplet of images for origin_name
        :param origin_name:
        :return: triplet where the pictures are: origin_image_path, classes_image_path (each class has its own color), objects_image_path (each object has its own color)
        """
        for i, val in enumerate(self.train):
            if has_origin(val, origin_name):
                filtered_files = list(filter(lambda name: has_origin(name, origin_name), self.train))

                if len(filtered_files) >= 3:
                    return filtered_files[0], filtered_files[1], filtered_files[2]
                elif len(filtered_files) >= 1:
                    return filtered_files[0], filtered_files[0], filtered_files[0]

    def get_iterable_origins(self, data):
        """
        Returns pure list of file names for which we can iterate and get the files by get_triple
        :return:
        """
        set_to_ret = set({})
        for i, val in enumerate(data):
            set_to_ret.add(get_origin_name(val))

        return list(set_to_ret)

    def get_triplet(self, origin_name, data):
        """
        Returns the triplet of images for origin_name
        :param origin_name: origin name
        :param data: structure which should be discovered
        :return: triplet where the pictures are: origin_image_path, classes_image_path (each class has its own color), objects_image_path (each object has its own color)
        """
        for i, val in enumerate(data):
            if has_origin(val, origin_name):
                filtered_files = list(filter(lambda name: has_origin(name, origin_name), data))

                if len(filtered_files) >= 3:
                    return filtered_files[0], filtered_files[1], filtered_files[2]
                elif len(filtered_files) >= 1:
                    return filtered_files[0], filtered_files[0], filtered_files[0]
        return None, None, None

    def get_triplets_with_class_template(self, data):
        """
        This is a list of records (triplet: origin_image_path, classes_image_path, objects_image_path).
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
        files = []
        for _, origin in enumerate(self.get_iterable_origins(data)):
            origin_img, class_img, object_img = self.get_triplet(origin, data)
            if origin_img == class_img:
                continue
            else:
                files.append((origin_img, class_img, object_img))  # append triplet because it contains class template

        return files

    def triplets_to_array(self, triplets):
        """
        Unpack triplets into 1D array
        :param triplets: list of records (origin_image_path, classes_image_path, objects_image_path)
        :return : array of strings
        """
        data = []
        for _, triplet in enumerate(triplets):
            origin_image_path, classes_image_path, objects_image_path = triplet
            data.append(origin_image_path)
            data.append(classes_image_path)
            data.append(objects_image_path)

        return data

    def split_couch(self, ratio, shuffle_flag=False):
        """
        Splits couch into 2 couches with different datasets
        For train and val parts it packed them into triplets (when shuffling is ON it would throw many class
        templates away), after splitting, they are unpacked
        Files without
        :ratio: <0,1.0> ratio of datasets length
        :return: Couch, Couch => Couch1 and Couch2 (with its datasets)
        """
        test1, test2 = split_data(self.test, ratio, shuffle_flag)
        train1, train2 = split_data(self.get_triplets_with_class_template(self.train), ratio, shuffle_flag)
        val1, val2 = split_data(self.get_triplets_with_class_template(self.val), ratio, shuffle_flag)

        dataset1 = Couch(self.dataset)
        dataset2 = Couch(self.dataset)

        dataset1.test = test1
        dataset1.train = self.triplets_to_array(train1)
        dataset1.val = self.triplets_to_array(val1)

        dataset2.test = test2
        dataset2.train = self.triplets_to_array(train2)
        dataset2.val = self.triplets_to_array(val2)

        return dataset1, dataset2


if __name__ == '__main__':
    couch = Couch(directory)
    c1, c2 = couch.split_couch(0.2)
    print(couch.val)
    print(c1.val)
    print(c2.val)
    s_example, s_reference, s_to_compare = couch.get_train_triple(couch.get_iterable_trains()[0])
    print(s_example, s_reference, s_to_compare)
    cropper = Cropper()
    cropper.set_imgs(s_example, s_reference, s_to_compare)
    example, reference, reference_boundary = cropper.next_crop()

    # result = model.runComputation(example)
    result = reference

    num_rows, num_cols = reference.shape[:2]
    translation_matrix = np.float32([[1, 0, 1], [0, 1, 0]])
    img_translation = cv2.warpAffine(reference, translation_matrix, (num_cols, num_rows))

    score, ref, res, diff, thresh = get_imgs_difference_structural_similarity(reference, img_translation)
    # cv2.imshow("Example", example)
    # cv2.imshow("Reference", ref)
    # cv2.imshow("Modified", res)
    # cv2.imshow("Diff", diff)
    # cv2.imshow("Thresh", thresh)
    print("Score get_imgs_difference_structural_similarity: {}".format(score))
    #im_h = cv2.hconcat([example, reference, reference_boundary])
    #cv2.imshow("All", im_h)
    #im_h = cv2.hconcat([thresh, diff])
    #cv2.imshow("Thresh and diff", im_h)
    #print(get_classes_in_image(reference))

    im_h = cv2.hconcat([reference, img_translation])
    cv2.imshow("Translation", im_h)
    print("Score get_imgs_difference_iou: {}".format(get_imgs_difference_iou(reference, img_translation)))
    cv2.waitKey(0)

    # for i, img in enumerate(os.listdir("results")):
    #     if "_target.png" in img:
    #         generated = img.split("_target.png")[0]
    #         cv_gen = cv2.imread(os.path.join("results", generated + ".png"))
    #         cv_tar = cv2.imread(os.path.join("results", img))
    #         print("Score get_imgs_difference_iou: {}".format(get_imgs_difference_iou(cv_gen, cv_tar)))
    #         score, ref, res, diff, thresh = get_imgs_difference_structural_similarity(reference, img_translation)
    #         print("Score get_imgs_difference_structural_similarity: {}".format(score))

