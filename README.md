## Neural Style Transfer in FastAI
Implementing ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/pdf/1603.08155.pdf) with FastAI framework. Apply hook (PyTorch mechanism) to calculate loss.


## Reference
1. [fastai -- Extracting Intermediate Features Using Forward Hook](https://github.com/TheShadow29/FAI-notes/blob/master/notebooks/Using-Forward-Hook-To-Save-Features.ipynb)
2. [fastai -- 06_cuda_cnn_hooks_init.ipynb, Deep Learning for Coder part 2 v3](https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb)
3. [fastai -- documentation on what a callback can unpack from kwargs](https://docs.fast.ai/callback.html)
4. [pytorch -- fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style)
5. [arxiv -- Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
6. [arxiv -- Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
7. [medium -- 10 Useful ML Practices For Python Developers](https://medium.com/modern-nlp/10-great-ml-practices-for-python-developers-b089eefc18fc)

## Log
#### [02/04/2020]
- need to integrate torch Dataset into fastai framework
- need to test MetaModel class, whether the hook result change with input feed
  
#### [03/04/2020]
- left with dataset definition

#### [04/04/2020]
- write hook in callbacks later