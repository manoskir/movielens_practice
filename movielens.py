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

'''
looking into the features from the dataset to understand what we're dealing with and how we should preprocess them - uncomment to use
from pprint import pprint

for rating in ratings_trainset.take(1).as_numpy_iterator():
	pprint(rating)
'''

#normalizing numerical features
#making a Keras Normalization layer to standardize a numerical feature
timestamp_normalization_layer = \
	tf.keras.layers.experimental.preprocessing.Normalization(axis=None)

# Normalization layer is a non-trainable layer and its state (mean and std of
# feature set) must be set before training in a step called "adaptation".
timestamp_normalization_layer.adapt(
	ratings_trainset.map(
		lambda x: x['timestamp']
		)
	)

for rating in ratings_trainset.take(3).as_numpy_iterator():
	print(
		f"Raw timestamp: {rating['timestamp']} ->",
		f"Normalized timestamp: {timestamp_normalization_layer(rating['timestamp'])}"
		)


#normalizing categorical features (set of fixed values), turning them into a high-dimensional embedding vector. the first step is to build a mapping (vocabulary) that maps each raw values and then turn them into embedding vectors
#making a Keras StringLookup layer as the mapping (lookup)
user_id_lookup_layer = \
	tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)

# StringLookup layer is a non-trainable layer and its state (the vocabulary)
# must be constructed and set before training in a step called "adaptation".
user_id_lookup_layer.adapt(
	ratings_trainset.map(
		lambda x: x['user_id']
		)
	)

print(
    f"Vocabulary[:10] -> {user_id_lookup_layer.get_vocabulary()[:10]}"
    # Vocabulary: ['[UNK]', '405', '655', '13', ...]
    # The vocabulary includes one (or more!) unknown (or "out of vocabulary", OOV)
    # tokens. So the layer can handle categorical values that are not in the
    # vocabulary and the model can continue to learn about and make
    # recommendations even using features that have not been seen during
    # vocabulary construction.
)

print(
    "Mapped integer for user ids: ['-2', '13', '655', 'xxx']\n",
    user_id_lookup_layer(
        ['-2', '13', '655', 'xxx']
    )
)

user_id_embedding_dim = 32
# The larger it is, the higher the capacity of the model, but the slower it is
# to fit and serve and more prone to overfitting.

user_id_embedding_layer = tf.keras.layers.Embedding(
	#Size of the vocabulary
	input_dim = user_id_lookup_layer.vocab_size(),
	#Dimension of the dense embedding
	output_dim = user_id_embedding_dim
	)

# A model that takes raw string feature values (user_id) in and yields embeddings
user_id_model = tf.keras.Sequential(
	[
	user_id_lookup_layer,
	user_id_embedding_layer
	]
	)

print(
    "Embeddings for user ids: ['-2', '13', '655', 'xxx']\n",
    user_id_model(
        ['-2', '13', '655', 'xxx']
    )
)

movie_id_lookup_layer = \
    tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
movie_id_lookup_layer.adapt(
    ratings_trainset.map(
        lambda x: x['movie_id']
    )
)

# Same as user_id_embedding_dim to be able to measure the similarity
movie_id_embedding_dim = 32

movie_id_embedding_layer = tf.keras.layers.Embedding(
    input_dim=movie_id_lookup_layer.vocab_size(),
    output_dim=movie_id_embedding_dim
)
 
movie_id_model = tf.keras.Sequential(
    [
        movie_id_lookup_layer,
        movie_id_embedding_layer
    ]
)

print(
    f"Embedding for the movie 898:\n {movie_id_model('898')}"
)


#tokenization of textual features and translation into embeddings - good candidates are cold-start or long-tail scenarios. In this instance we don't have comments etc, but movie titles with similar titles are likely to belong in the same series (e.g.Harry Potter)

# Keras TextVectorization layer transforms the raw texts into `word pieces` and
# map these pieces into tokens.
with tf.device('/CPU:0'): # for this particular line, there's an issue with Apple devices and I need to explicitly specify to use CPU to make it work or else it crashes.
	movie_title_vectorization_layer = \
		tf.keras.layers.experimental.preprocessing.TextVectorization()

	movie_title_vectorization_layer.adapt(
		ratings_trainset.map(
			lambda rating: rating['movie_title']
			)
		)

# Verify that the tokenization is done correctly
print(
    "Vocabulary[40:50] -> ",
    movie_title_vectorization_layer.get_vocabulary()[40:50]
)

print(
    "Vectorized title for 'Postman, The (1997)'\n",
    movie_title_vectorization_layer('Postman, The (1997)')
)

movie_title_embedding_dim = 32
movie_title_embedding_layer = tf.keras.layers.Embedding(
    input_dim=len(movie_title_vectorization_layer.get_vocabulary()),
    output_dim=movie_title_embedding_dim,
    # Whether or not the input value 0 is a MASK token.
    # Keras TextVectorization layer builds the vocabulary with MASK token.
    mask_zero=True
)

movie_title_model = tf.keras.Sequential(
    [
       movie_title_vectorization_layer,
       movie_title_embedding_layer,
       # each title contains multiple words, so we will get multiple embeddings
       # for each title that should be compressed into a single embedding for
       # the text. Models like RNNs, Transformers or Attentions are useful here.
       # However, averaging all the words' embeddings together is also a good
       # starting point.
       tf.keras.layers.GlobalAveragePooling1D()
    ]
)
