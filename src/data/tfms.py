from torchvision import transforms

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s  %(name)s  %(levelname)s  %(message)s")


def get_transforms(img_size):
    """ 
    transform input data

    :param:
        img_size : target resize (h, w) 
    """
    tfms_ls = [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255.))
    ]
    logging.info(f'train data transform is initialized')
    return transforms.Compose(tfms_ls)


def get_test_transforms():
    tfms_ls = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255.))
        ]
    logging.info(f'test data transform is initialized')
    return transforms.Compose(tfms_ls)


def get_style_transforms(img_size):
    tfms_ls = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ]
    return transforms.Compose(tfms_ls)
