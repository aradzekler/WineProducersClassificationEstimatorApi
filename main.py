import tensorflow as tf
from sklearn.datasets import load_wine  # importing our dataset
from sklearn.model_selection import train_test_split  # for splitting the dataset into train and test sets.
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


wine_data = load_wine()  # and loading it.
print(wine_data['DESCR'])

'''
 the data for every feature (13) for every wine,
 such as alcohol, ash, magnesium.. labeled as data
 for this is the raw data of the wine (his properties)
'''
feature_data = wine_data['data']

'''
3 classes of wine (3 different cultivators), 
this is labeled as target because that's what
we are trying to predict.
'''
label_data = wine_data['target']

X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data
                                                    , test_size=0.3)

scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# a nice way to create a feature column
feature_cols = [tf.feature_column.numeric_column('x', shape=[13])]

'''
A dense deep model with 3 hidden layers of 13 neurons each
and a Gradient Descent Optimizer.
'''
deep_model = tf.estimator.DNNClassifier(hidden_units=[13, 13, 13],
                                        feature_columns=feature_cols,
                                        n_classes=3,
                                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))

input_fun = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': scaled_X_train},
                                                         y=y_train,
                                                         shuffle=True,
                                                         batch_size=10,
                                                         num_epochs=5)

deep_model.train(input_fn=input_fun, steps=500)

'''
Our evaluation function used to tell how good is our prediction, that
by testing (x_test data) an input in a serial order.
'''
input_fun_eval = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': scaled_X_test},
                                                              shuffle=False)

preds = list(deep_model.predict(input_fn=input_fun_eval))
print(preds)  # a messy looking dataset, luckily we can organize it easly!

# for an easier parsing by classification_report
predictions = [p['class_ids'][0] for p in preds]

print(classification_report(y_test, predictions))

