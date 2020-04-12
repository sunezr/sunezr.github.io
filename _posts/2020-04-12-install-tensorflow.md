
# Install tensorflow 

window install tensorflow(GPU unknow)

$ conda install -c conda-forge tensorflow 

reference:https://anaconda.org/conda-forge/tensorflow

https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/

error: https://github.com/tensorflow/tensorflow/issues/35749

1. https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads

2. download x64: vc_redist.x64.exe

3. install it.


```python
import tensorflow as tf
```


```python
tf.print("tensorflow version:", tf.__version__)
```

    tensorflow version: 2.1.0

