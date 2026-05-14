import os
import io
import logging
import warnings
from contextlib import redirect_stderr

# Suppress TensorFlow/Keras warnings BEFORE importing tensorflow/keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Python warnings from Keras and TensorFlow
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('keras').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*tf.function.*')

# Capture and discard stderr during keras imports
stderr_capture = io.StringIO()

with redirect_stderr(stderr_capture):
    from datetime import datetime, timedelta
    from random import choice
    import threading
    from time import sleep

    from statsmodels.tsa.arima.model import ARIMA
    from numpy import array, concatenate
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint

import workload, ram_usage


def naive(hostname):
    df = workload.get(hostname)
    last_time_stamp = df.tail(1).index
    if last_time_stamp > (datetime.now() - timedelta(seconds=90)):
        predict = df.iloc[-1]['mem']
    else:
        predict = ram_usage.get(hostname)
    return predict

def arima(hostname):
    df = workload.get(hostname)
    df = df.resample('s').interpolate().resample('90s').asfreq()
    order = (1,0,1)
    X = [x for x in df.mem]
    try:
        model = ARIMA(X, order=order, enforce_stationarity=False)
        model_fit = model.fit()
        predict = model_fit.forecast()[0]
    except ValueError:
        print('Insufficient data to use arima model.\nUsing default mode.')
        predict = ram_usage.get(hostname)
    return predict

def split_dataframes(df):
    df = df.reset_index()
    dataframes = [g.set_index('time_stamp', drop=True) for k,g  in df.groupby((~(df.time_stamp.diff().dt.total_seconds().fillna(0) < 240)).cumsum())]
    return dataframes

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def concatenate_samples(full_df, n_steps):
    Xs = []
    ys = []
    for df in split_dataframes(full_df):
        if len(df) > n_steps + 1:
            df_arr = df['mem'].values
            X, y = split_sequence(df_arr, n_steps)
            Xs.append(X)
            ys.append(y)
    Xsample = concatenate((Xs))
    ysample = concatenate((ys))
    
    return Xsample, ysample, df_arr

def loss_average(history, epochs):
    second_half_loss = history.history['loss'][epochs//2:]
    return sum(second_half_loss) / len(second_half_loss)

def train_lstm_model(hostname):
    try:
        epochs = choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        # get data workload
        df = workload.get(hostname)

        # choose a number of time steps
        n_steps = 50

        # split into samples
        X, y, df_arr = concatenate_samples(df, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # split train and test
        split_index = len(X)*2//3
        Xtrain, Xtest = X[:-split_index], X[-split_index:]
        ytrain, ytest = y[:-split_index], y[-split_index:]
        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # create a model check point
        filepath = f'models/{hostname}/weights-{epochs:02d}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callbacks_list = [checkpoint]
        # fit model
        history = model.fit(Xtrain, ytrain, epochs=epochs, verbose=0, validation_data=(Xtest, ytest),callbacks=callbacks_list)
        # save model named as loss value
        loss = loss_average(history, epochs)
        loss = '.'.join([str(loss).split('.')[0].zfill(3), str(loss).split('.')[-1]])
        time_stamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        model.load_weights(filepath)
        model.save(f'models/{hostname}/{loss}_{time_stamp}_{epochs}.keras')

        return model
    except Exception as e:
        print(f'Unable to train now: {e}')

def select_best_model(hostname):
    """Select best model based on actual loss value, not filename."""
    model_dir = f'./models/{hostname}'
    models = [f for f in os.listdir(model_dir) if f.endswith('.keras')]

    if not models:
        return None

    # Parse loss from filenames (format: {loss}_{timestamp}_{epochs}.keras)
    model_losses = []
    for model_file in models:
        try:
            loss = float(model_file.split('_')[0])
            model_losses.append((loss, model_file))
        except (ValueError, IndexError):
            continue

    if not model_losses:
        return None

    # Sort by loss and return best
    model_losses.sort(key=lambda x: x[0])
    best_model = model_losses[0][1]

    # Keep only 5 best models, delete rest
    for loss, worst_file in model_losses[5:]:
        try:
            os.remove(f'{model_dir}/{worst_file}')
        except OSError:
            pass

    # Clean up intermediate weight files
    for f in os.listdir(model_dir):
        if f.startswith('weights-'):
            try:
                os.remove(f'{model_dir}/{f}')
            except OSError:
                pass

    return best_model


class LSTMTrainingManager:
    """Manages asynchronous LSTM training for multiple hosts.
    Training runs continuously in background threads.
    Predictions never wait for training to complete.
    """

    def __init__(self):
        self.training_threads = {}
        self.model_cache = {}
        self.cache_lock = threading.Lock()
        self.stop_event = threading.Event()

    def start_training(self, hostname):
        """Start continuous training thread for a host."""
        if hostname in self.training_threads and self.training_threads[hostname].is_alive():
            return  # Already training

        os.makedirs(f'models/{hostname}', exist_ok=True)

        thread = threading.Thread(
            target=self._training_loop,
            args=[hostname],
            daemon=True
        )
        thread.start()
        self.training_threads[hostname] = thread

    def get_best_model(self, hostname):
        """Get best trained model without blocking."""
        # Check cache first
        with self.cache_lock:
            if hostname in self.model_cache:
                model, loss, timestamp = self.model_cache[hostname]
                if (datetime.now() - timestamp).total_seconds() < 3600:
                    return model
                else:
                    del self.model_cache[hostname]

        # Try loading from disk
        try:
            best_model_file = select_best_model(hostname)
            if best_model_file:
                model = load_model(f'models/{hostname}/{best_model_file}')
                loss = float(best_model_file.split('_')[0])

                with self.cache_lock:
                    self.model_cache[hostname] = (model, loss, datetime.now())
                return model
        except Exception:
            pass

        return None

    def stop_training(self, hostname=None):
        """Stop training thread(s). If hostname is None, stops all training threads."""
        if hostname:
            if hostname in self.training_threads:
                # Daemon threads will exit naturally
                self.training_threads.pop(hostname, None)
        else:
            self.stop_event.set()
            for thread in self.training_threads.values():
                if thread.is_alive():
                    thread.join(timeout=5)
            self.stop_event.clear()
            self.training_threads.clear()

    def _training_loop(self, hostname):
        """Continuous training loop running in background."""
        while not self.stop_event.is_set():
            try:
                df = workload.get(hostname)
                last_df = split_dataframes(df)[-1]

                if len(last_df) > 50 and len(df) > 100:
                    model = train_lstm_model(hostname)

                    if model:
                        # Validate prediction quality
                        n_steps = 50
                        x_input = array(df[-n_steps:].mem.values)
                        x_input = x_input.reshape((1, n_steps, 1))
                        predict = model.predict(x_input, verbose=0)[0][0]
                        actual = ram_usage.get(hostname)

                        if abs(predict - actual) < 15:
                            best_file = select_best_model(hostname)
                            if best_file:
                                loss = float(best_file.split('_')[0])
                                with self.cache_lock:
                                    self.model_cache[hostname] = (model, loss, datetime.now())

                sleep(60)  # Rate limit: max 1 train/minute/host

            except Exception as e:
                print(f'Training error for {hostname}: {e}')
                sleep(60)


def lstm(hostname):
    """
    Non-blocking LSTM prediction using best available model.
    Never blocks on training operations - uses cached model or falls back gracefully.
    """
    # Check if a trained model is available
    best_model = lstm_manager.get_best_model(hostname)

    if best_model:
        try:
            df = workload.get(hostname)
            n_steps = 50

            # Prepare input data (need at least 50 samples)
            if len(df) >= n_steps:
                n_features = 1
                x_input = array(df[-n_steps:].mem.values)
                x_input = x_input.reshape((1, n_steps, n_features))

                predict = best_model.predict(x_input, verbose=0)[0][0]
                print(f'LSTM prediction for {hostname}: {predict}')
                return predict
            else:
                print(f'Insufficient data for LSTM prediction on {hostname}: {len(df)} samples, using current RAM')
                return ram_usage.get(hostname)

        except (IndexError, ValueError, AttributeError) as e:
            print(f'LSTM prediction failed for {hostname}: {e}')
            return ram_usage.get(hostname)
    else:
        print(f'No trained model available for {hostname}, using current RAM')
        return ram_usage.get(hostname)


# Global training manager instance
lstm_manager = LSTMTrainingManager()
