

# Network building
net = tflearn.input_data([None, 200])
# create embedding weights, set trainable to False, so weights are not updated
net = tflearn.embedding(net, input_dim=20000, output_dim=128, trainable=False, name="EmbeddingLayer")
net = tflearn.lstm(net, 128)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

# Retrieve embedding layer weights (only a single weight matrix, so index is 0)
embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
# Assign your own weights (for example, a numpy array [input_dim, output_dim])
model.set_weights(embeddingWeights, YOUR_WEIGHTS)

# Train with your custom weights
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=128)