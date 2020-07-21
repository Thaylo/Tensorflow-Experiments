import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import random


def main():
    epochs = 180
    feature_columns = ['x', 'y']
    label_columns = ['z']
    columns = feature_columns + label_columns
    features, labels = generate_dataset(columns)
    features = norm(features, feature_columns)
    model = create_model(features, labels)
    model.fit(features, labels, epochs=epochs, verbose=1)
    display_comparison(features, labels, model)


def create_model(features, labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='relu', input_shape=[len(features.keys())]),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(len(labels.keys()), activation='linear')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse', 'mae'])
    return model


def display_comparison(features, labels, model):
    example_labels = labels[:10]
    example_features = features[:10]
    example_result = model.predict(example_features)
    print("Expected | Predicted")
    # iterating over rows using iterrows() function
    for i, j in example_labels.iterrows():
        print(j[0], example_result[i][0])


def generate_dataset(columns):
    rows_list = []
    for i in range(400):
        new_line = dict()
        a = random.random()+1
        b = random.random()+1
        new_line['x'] = a
        new_line['y'] = b
        new_line['z'] = a * b
        rows_list.append(new_line)
    dataset = pd.DataFrame(rows_list, columns=['x', 'y', 'z'])
    print(dataset)
    features = dataset[['x', 'y']]
    labels = dataset[['z']]
    return features, labels


def norm(df, columns):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=columns)


if __name__ == "__main__":
    main()
