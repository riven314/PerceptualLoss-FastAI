## Neural Style Transfer in FastAI
Implementing ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/pdf/1603.08155.pdf) with FastAI framework. Apply hook (PyTorch mechanism) to calculate loss.


## Reference
1. [fastai -- Extracting Intermediate Features Using Forward Hook](https://github.com/TheShadow29/FAI-notes/blob/master/notebooks/Using-Forward-Hook-To-Save-Features.ipynb)
2. [fastai -- 06_cuda_cnn_hooks_init.ipynb, Deep Learning for Coder part 2 v3](https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb)
3. [pytorch -- fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style)
4. [arxiv -- Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
5. [arxiv -- Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
6. [medium -- 10 Useful ML Practices For Python Developers](https://medium.com/modern-nlp/10-great-ml-practices-for-python-developers-b089eefc18fc)

## Log
### [01/04/2020]
- need to integrate torch Dataset into fastai framework
- need to test MetaModel class, whether the hook result change with input feed