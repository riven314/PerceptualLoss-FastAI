from torchvision import transforms


def get_transforms(img_size):
    """ img_size : target resize (h, w) """
    tfms_ls = [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255.))
    ]
    return transforms.Compose(tfms_ls)
