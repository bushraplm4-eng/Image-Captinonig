import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# 1. ENCODER: Feature Extraction using ResNet50
def build_encoder():
    # Load ResNet50 pre-trained on ImageNet
    base_model = ResNet50(weights='imagenet')
    # Remove the last classification layer to get the feature vector
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

def extract_features(image_path, model):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = model.predict(img, verbose=0)
    return feature

# 2. DECODER: Generating Captions using LSTM
def build_caption_model(vocab_size, max_length):
    # Image feature input (from ResNet50)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence/Text input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder (Merging both)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# 3. INFERENCE: Generating the caption
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo_features, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word
        yhat = model.predict([photo_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# --- SETUP NOTE ---
# In a real scenario, you would:
# 1. Load a 'tokenizer.pkl' file (created during training on Flickr8k)
# 2. Load 'model_weights.h5' (the trained weights)
print("Model Architecture Initialized. Ready for training or loading weights.")