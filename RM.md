### Title:  Tensorboard + Keras

I am very interested in the visualization part of this exercise. The tutorial url: https://www.youtube.com/playlist?list=PLX-LrBk6h3wR27xylD3Rsx4bbA15jlcYC

**I will simply introduce the Tensorboard visualization tool and show some basic operation mixed with keras as shown in the code.**



#### *1： Get the train/test datasets*

In the tutorial, Mark gives a kaggle url: https://www.kaggle.com/zalando-research/fashionmnist

We download the dataset into a 'data' folder.



#### *2: import Tensorflow package*

In the 2nd part of the exercise, we come to Tensorboard for the first time, and we import the package and generate the corresponding file called 'events.out.tfevents....'.

`from keras.callbacks import TensorBoard`

...

`tensorboard = TensorBoard(`

`#define Tensorflow object attributes`

​    `log_dir=r'logs/{}'.format('cnn_1layer'),`
    `write_graph=True,`
    `write_grads=True,`
    `histogram_freq=1,`
    `write_images=True`

`)`

...

`cnn_model.fit(`
    `...`
    `callbacks=[tensorboard]`

`)`

After we trained the model, we get the entire 'event' file, and then we run the following code in the terminal:

`tensorboard --logdir=path/to/log-directory`

'logdir' is the path variable we defined in the Tensorboard object. 

![1535722967003](C:\Users\11198\AppData\Local\Temp\1535722967003.png)

We go to the site: http://laptop-t8vk69p8:6006 as shown in the terminal. Below is what tools we could use as defined in the Tensorflow object.

![1535722903837](C:\Users\11198\AppData\Local\Temp\1535722903837.png)



#### *3: use the Embedding tool and view the date*

In the 3rd part of the exercise, we introduce some tensorflow package and need a few steps to realize the 'Projector' function.

##### Basic workflow

- Read the Fahsion MNIST data and create an X (image) and Y (label) batch
- Create a Summary Writer
- Create the embedding Tensor from X
- Configure the projector
- Run the TF session and create a model check-point
- create the sprite image
- create the metadata (labels) file



Firstly, we import the relevant package:

```
import ...
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
```



Define the log directory:

`logdir = r'C:\Users\Fashion_MNIST\logdir'`



Setup the write and embedding tensor

```python
# setup the write and embedding tensor

summary_writer = tf.summary.FileWriter(logdir)

embedding_var = tf.Variable(x_test, name='fmnist_embedding')

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')
embedding.sprite.single_image_dim.extend([28, 28])

projector.visualize_embeddings(summary_writer, config)
```

Run the sesion to create the model check point

```python
# run the sesion to create the model check point

with tf.Session() as sesh:
    sesh.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sesh, os.path.join(logdir, 'model.ckpt'))
```

Create the sprite image and the metadata file

```python
# create the sprite image and the metadata file

rows = 28
cols = 28

label = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat',
          'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

sprite_dim = int(np.sqrt(x_test.shape[0]))

sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim))

index = 0
labels = []
for i in range(sprite_dim):
    for j in range(sprite_dim):
        
        labels.append(label[int(y_test[index])])
        
        sprite_image[
            i * cols: (i + 1) * cols,
            j * rows: (j + 1) * rows
        ] = x_test[index].reshape(28, 28) * -1 + 1
        
        index += 1
        
with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        meta.write('{}\t{}\n'.format(index, label))
        
plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')
plt.show()
```

After running the upper code, we run

`tensorboard --logdir=path/to/log-directory`

'logdir' is the path variable we defined in the Tensorboard object. 

![1535722967003](C:\Users\11198\AppData\Local\Temp\1535722967003.png)

We go to the site: http://laptop-t8vk69p8:6006 as shown in the terminal. 

Then we could get one more tool:

![1535724042210](C:\Users\11198\AppData\Local\Temp\1535724042210.png)

We could use T-SNE or PCA to actually view the data which is coolest part I think in this tutorial. 