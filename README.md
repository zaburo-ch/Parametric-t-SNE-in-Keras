Parametric t-SNE  
====
An implementation of ["Parametric t-SNE"](https://lvdmaaten.github.io/publications/papers/AISTATS_2009.pdf) in Keras.  
Authors used stacked RBM in the paper, but we used simple ReLU units instead.  

I used [the python implementation of t-SNE](https://lvdmaaten.github.io/tsne/code/tsne_python.zip) by [Laurens van der Maaten](https://lvdmaaten.github.io/) as a reference.  

For some reason, this code is not working on Keras 1.0.4.  
So we should use Keras 1.0.3.  
```pip install Keras==1.0.3```  
