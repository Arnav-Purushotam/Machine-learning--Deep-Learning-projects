#feature extraction from images


import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model

# load the ResNet50 Model
feature_extractor = ResNet50(weights='imagenet', include_top=False)
feature_extractor_new = Model(feature_extractor.input, feature_extractor.layers[-2].output)
feature_extractor_new.summary()

for file in os.listdir(image_path):
    path = image_path + "//" + file
    img = image.load_img(path, target_size=(90, 90))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    feature = feature_extractor_new.predict(img_data)
    feature_reshaped = np.array(feature).flatten()


#processing captions


import pandas as pd


captions = pd.read_csv('/kaggle/input/flickr8k/captions.txt', sep=",")
captions = captions.rename(columns=lambda x: x.strip().lower())
captions['image'] = captions['image'].apply(lambda x: x.split(".")[0])
captions = captions[['image', 'caption']]

captions['caption'] = "<start> " + captions['caption'] + " <end>"

# in case we have any missing caption/blank caption drop it
print(captions.shape)
captions = captions.dropna()
print(captions.shape)

# training and testing image captions split
train_image_captions = {}
test_image_captions = {}


all_captions = []


for image in train_data_images:
    tempDf = captions[captions['image'] == image]
    list_of_captions = tempDf['caption'].tolist()
    train_image_captions[image] = list_of_captions
    all_captions.append(list_of_captions)


for image in test_data_images:
    tempDf = captions[captions['image'] == image]
    list_of_captions = tempDf['caption'].tolist()
    test_image_captions[image] = list_of_captions
    all_captions.append(list_of_captions)

print("Data Statistics")
print(f"Training Images Captions {len(train_image_captions.keys())}")
print(f"Testing Images Captions {len(test_image_captions.keys())}")

#tokenizing captions


import spacy

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

# tokenize evry captions, remove punctuations, lowercase everything
for key, value in train_image_captions.items():
    ls = []
    for v in value:
        doc = nlp(v)
        new_v = " "
        for token in doc:
            if not token.is_punct:
                if token.text not in [" ", "\n", "\n\n"]:
                    new_v = new_v + " " + token.text.lower()

        new_v = new_v.strip()
        ls.append(new_v)
    train_image_captions[key] = ls


all_captions = [caption for list_of_captions in all_captions for caption in list_of_captions]

# use spacy to convert to lowercase and reject any special characters
tokens = []
for captions in all_captions:
    doc = nlp(captions)
    for token in doc:
        if not token.is_punct:
            if token.text not in [" ", "\n", "\n\n"]:
                tokens.append(token.text.lower())


import collections

word_count_dict = collections.Counter(tokens)
reject_words = []
for key, value in word_count_dict.items():
    if value < 10:
        reject_words.append(key)

reject_words.append("<")
reject_words.append(">")

# remove tokens that are in reject words
tokens = [x for x in tokens if x not in reject_words]

# convert the token to equivalent index using Tokenizer class of Keras
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)


# compute length of vocabulary and maximum length of a caption (for padding)
vocab_len = len(tokenizer.word_counts) + 1
print(f"Vocabulary length - {vocab_len}")

max_caption_len = max([len(x.split(" ")) for x in all_captions])
print(f"Maximum length of caption - {max_caption_len}")

#creating training dataset

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical



def create_trianing_data(captions, images, tokenizer, max_caption_length, vocab_len, photos_per_batch):
    X1, X2, y = list(), list(), list()
    n = 0


    while 1:
        for key, cap in captions.items():
            n += 1

            image = images[key]

            for c in cap:

                sequnece = [tokenizer.word_index[word] for word in c.split(' ') if
                            word in list(tokenizer.word_index.keys())]



                for i in range(1, len(sequence)):

                    inp, out = sequence[:i], sequence[i]

                    input_seq = pad_sequences([inp], maxlen=max_caption_length)[0]

                    output_seq = to_categorical([out], num_classes=vocab_len)[0]

                    X1.append(image)
                    X2.append(input_seq)
                    y.append(output_seq)


            if n == photos_per_batch:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0

#creating the final model

import keras


def create_model(max_caption_length, vocab_length):
    # sub network for handling the image feature part
    input_layer1 = keras.Input(shape=(18432))
    feature1 = keras.layers.Dropout(0.2)(input_layer1)
    feature2 = keras.layers.Dense(max_caption_length * 4, activation='relu')(feature1)
    feature3 = keras.layers.Dense(max_caption_length * 4, activation='relu')(feature2)
    feature4 = keras.layers.Dense(max_caption_length * 4, activation='relu')(feature3)
    feature5 = keras.layers.Dense(max_caption_length * 4, activation='relu')(feature4)

    # sub network for handling the text generation part
    input_layer2 = keras.Input(shape=(max_caption_length,))
    cap_layer1 = keras.layers.Embedding(vocab_length, 300, input_length=max_caption_length)(input_layer2)
    cap_layer2 = keras.layers.Dropout(0.2)(cap_layer1)
    cap_layer3 = keras.layers.LSTM(max_caption_length * 4, activation='relu', return_sequences=True)(cap_layer2)
    cap_layer4 = keras.layers.LSTM(max_caption_length * 4, activation='relu', return_sequences=True)(cap_layer3)
    cap_layer5 = keras.layers.LSTM(max_caption_length * 4, activation='relu', return_sequences=True)(cap_layer4)
    cap_layer6 = keras.layers.LSTM(max_caption_length * 4, activation='relu')(cap_layer5)

    # merging the two sub network
    decoder1 = keras.layers.merge.add([feature5, cap_layer6])
    decoder2 = keras.layers.Dense(256, activation='relu')(decoder1)
    decoder3 = keras.layers.Dense(256, activation='relu')(decoder2)

    # output is the next word in sequence
    output_layer = keras.layers.Dense(vocab_length, activation='softmax')(decoder3)
    model = keras.models.Model(inputs=[input_layer1, input_layer2], outputs=output_layer)

    model.summary()

    return model


#creating word embeddings



import spacy

nlp = spacy.load('en_core_web_lg')


embedding_dimension = 300
embedding_matrix = np.zeros((vocab_len, embedding_dimension))


for word, index in tokenizer.word_index.items():
    doc = nlp(word)
    embedding_vector = np.array(doc.vector)
    embedding_matrix[index] = embedding_vector


predictive_model.layers[2]
predictive_model.layers[2].set_weights([embedding_matrix])
predictive_model.layers[2].trainable = False


#training the model


train_data = create_trianing_data(train_image_captions, train_image_features, tokenizer, max_caption_len, vocab_length, 32)


model = create_model(max_caption_len, vocab_len)

steps_per_epochs = len(train_image_captions)//32


model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit_generator(train_data, epochs=100, steps_per_epoch=steps_per_epochs)


#generation of captions

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
% matplotlib inline



def generate_captions(model, image, tokenizer.word_index, max_caption_length, tokenizer.index_word):

# input is <start>
input_text = '<start>'

# keep generating words till we have encountered <end>
for i in range(max_caption_length):
    seq = [tokenizer.word_index[w] for w in in_text.split() if w in list(tokenizer.word_index.keys())]
    seq = pad_sequences([sequence], maxlen=max_caption_length)
    prediction = model.predict([photo, sequence], verbose=0)
    prediction = np.argmax(prediction)
    word = tokenizer.index_word[prediction]
    input_text += ' ' + word
    if word == '<end>':
        break

# remove <start> and <end> from output and return string
output = in_text.split()
output = output[1:-1]
output = ' '.join(output)
return output


count = 0
for key, value in test_image_features.items():
    test_image = test_image_features[key]
    test_image = np.expand_dims(test_image, axis=0)
    final_caption = generate_captions(predictive_model, test_image, tokenizer.word_index, max_caption_len,
                                      tokenizer.index_word)

    plt.figure(figsize=(7, 7))
    image = Image.open(image_path + "//" + key + ".jpg")
    plt.imshow(image)
    plt.title(final_caption)

    count = count + 1
    if count == 3:
        break





