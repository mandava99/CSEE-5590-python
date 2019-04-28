import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import re
from joblib import parallel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


def createmodel():
    model = Sequential()
    model.add(Embedding(2000, 128, input_length=28))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data = pd.read_csv('Sentiment.csv')
    # Keeping only the neccessary columns
    data = data[['text', 'sentiment']]
    data = data[data.sentiment != 'Neutral']

    data = data[data.sentiment != "Neutral"]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    print(data[data['sentiment'] == 'Positive'].size)
    print(data[data['sentiment'] == 'Negative'].size)

    for idx, row in data.iterrows():
        row[0] = row[0].replace('rt', ' ')

    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)
    embed_dim = 128
    lstm_out = 196
    labelencoder = LabelEncoder()
    integer_encoded = labelencoder.fit_transform(data['sentiment'])
    y = to_categorical(integer_encoded)
    X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
    model1 = KerasClassifier(build_fn=createmodel, verbose=0)
    batch_size =[20, 30, 40]
    epochs =[3, 4, 5]
    param_grid= dict(batch_size=batch_size, epochs=epochs)
    grid=GridSearchCV(estimator=model1, param_grid=param_grid, n_jobs=-1)
    grid_result=grid.fit(X_train, Y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))