import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

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

# Use `tfds.as_dataframe()` to convert `tf.data.Dataset` to `pandas.DataFrame`.
# Add the `tfds.core.DatasetInfo` as second argument of `tfds.as_dataframe` to
# visualize images, audio, texts, videos, etc. `pandas.DataFrame` will load the
# full dataset in-memory, and can be very expensive to display. So use it only
# with take() function.

print(tfds.as_dataframe(ratings_dataset.take(5), ratings_dataset_info))

## Split dataset randomly (80% for train and 20% for test)

trainset_size = 0.8 * ratings_dataset.__len__().numpy()

# In an industrial recommender system, this would most likely be done by time:
# The data up to time T would be used to predict interactions after T.

# set the global seed:
tf.random.set_seed(42)

# Shuffle the elements of the dataset randomly.
ratings_dataset_shuffled = ratings_dataset.shuffle(
    # the new dataset will be sampled from a buffer window of first `buffer_size`
    # elements of the dataset
	buffer_size=100_000,
    # set the random seed that will be used to create the distribution.
	seed=42,
 # `list(dataset.as_numpy_iterator()` yields different result for each call
    # Because reshuffle_each_iteration defaults to True.
	reshuffle_each_iteration = False)

ratings_trainset = ratings_dataset_shuffled.take(trainset_size)
ratings_testset = ratings_dataset_shuffled.skip(trainset_size)

print("ratings_trainset size: %d" % ratings_trainset.__len__())
print("ratings_testset size: %d" % ratings_testset.__len__())