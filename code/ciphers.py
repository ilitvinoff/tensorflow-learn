import tensorflow as tf

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_test.shape: {y_test.shape}")