""" LSTM MODEL IMPLEMENTATION """

# create sequence for learning
def create_sequences(X, y, time_steps=72):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i+time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 72
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

# train/val/test split
n = len(X_seq)
train_size = int(n * 0.7)
val_size = int(n * 0.15)

X_train, y_train = X_seq[:train_size], y_seq[:train_size]
X_val, y_val = X_seq[train_size:train_size + val_size], y_seq[train_size:train_size + val_size]
X_test, y_test = X_seq[train_size + val_size:], y_seq[train_size + val_size:]

# define model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2), input_shape=(TIME_STEPS, X_seq.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# define callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
