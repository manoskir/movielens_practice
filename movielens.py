import tensorflow as tf
import tensorflow_datasets as tfds

# Download the data, save them as `tfrecord` files, load the `tfrecord` files
# and create the `tf.data.Dataset` object containing the dataset.

ratings_dataset, ratings_dataset_info = tfds.load(
	name = 'movielens/100k-ratings',
	# MovieLens dataset is not splitted into `train` and `test` sets by default.
    # So TFDS has put it all into `train` split. We load it completely and split
    # it manually.
	split = 'train',
	# `with_info=True` makes the `load` function return a `tfds.core.DatasetInfo`
    # object containing dataset metadata like version, description, homepage,
    # citation, etc.
	with_info = True
	)

# Calling the `tfds.load()` function in old versions of TFDS won't return an
# instance of `tf.data.Dataset` type. So we can make sure about it.

assert isinstance(ratings_dataset, tf.data.Dataset)

print("ratings_dataset_size: %d" % ratings_dataset.__len__())