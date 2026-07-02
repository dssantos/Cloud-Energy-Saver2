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
    from keras.layers import LSTM, Dense, Dropout
    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint
    import keras.optimizers

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

def split_sequence(sequence, n_steps, steps_ahead=1):
    """
    Split sequence into input/output samples.
    steps_ahead: How many steps ahead to predict (1=next, 6=3min)
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # find the target position (steps ahead)
        target_ix = end_ix + steps_ahead - 1
        # check if we are beyond the sequence
        if target_ix >= len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[target_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def concatenate_samples(full_df, n_steps, steps_ahead=1):
    """Concatenate samples with configurable prediction horizon."""
    Xs = []
    ys = []
    for df in split_dataframes(full_df):
        # Need enough data: n_steps input + steps_ahead target
        min_length = n_steps + steps_ahead
        if len(df) > min_length:
            df_arr = df['mem'].values
            X, y = split_sequence(df_arr, n_steps, steps_ahead)
            Xs.append(X)
            ys.append(y)
    Xsample = concatenate((Xs))
    ysample = concatenate((ys))

    return Xsample, ysample, df_arr

def loss_average(history, epochs):
    second_half_loss = history.history['loss'][epochs//2:]
    return sum(second_half_loss) / len(second_half_loss)

def random_hyperparameters():
    """Generate random hyperparameters for LSTM training."""
    return {
        'n_steps': choice([10, 15, 20, 30, 50]),  # Focus on recent data
        'lstm_units': choice([64, 128, 256]),
        'epochs': choice([50, 100, 150, 200, 250, 300, 400]),
        'batch_size': choice([8, 16, 32, 64]),
        'dropout': choice([0.1, 0.2, 0.5]),
        'learning_rate': choice([0.001, 0.002, 0.005])
    }

def update_model_loss(hostname, model_file, actual_loss):
    """
    Update model filename with new calculated loss.
    Renames model to reflect actual prediction performance and updates timestamp.
    """
    if not model_file:
        return None

    model_dir = f'models/{hostname}'
    old_path = f'{model_dir}/{model_file}'

    try:
        # Parse existing filename to get timestamp and hyperparameters
        # Format: {val_loss}_{timestamp}_{epochs}_{n_steps}_{units}_{steps_ahead}ahead.keras
        parts = model_file.replace('.keras', '').split('_')

        if len(parts) < 4:
            # Old format or invalid, skip
            return None

        # Update loss with new value
        new_loss_str = str(actual_loss)
        new_loss_str = '.'.join([new_loss_str.split('.')[0].zfill(3),
                                  new_loss_str.split('.')[-1][:6]])

        # Update timestamp to current time
        new_timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

        # Reconstruct filename with new loss and updated timestamp
        # Handle both old format (5 parts) and new format (6 parts with steps_ahead)
        if len(parts) >= 6 and parts[5].endswith('ahead'):
            steps_ahead = parts[5].replace('ahead', '')
            new_filename = f'{new_loss_str}_{new_timestamp}_{parts[2]}_{parts[3]}_{parts[4]}_{steps_ahead}ahead.keras'
        else:
            # Old format without steps_ahead
            new_filename = f'{new_loss_str}_{new_timestamp}_{parts[2]}_{parts[3]}_{parts[4]}.keras'

        new_path = f'{model_dir}/{new_filename}'

        # Rename the model file
        os.rename(old_path, new_path)
        print(f'[MODEL UPDATE] {hostname}: Renamed {model_file} -> {new_filename} (new loss: {actual_loss:.6f}, updated timestamp)')

        return new_filename
    except Exception as e:
        print(f'[ERROR] Failed to rename model {model_file}: {e}')
        return None

def train_lstm_model(hostname, steps_ahead=6):
    """Train LSTM model with configurable prediction horizon."""
    try:
        # Get random hyperparameters
        hp = random_hyperparameters()
        n_steps = hp['n_steps']
        lstm_units = hp['lstm_units']
        epochs = hp['epochs']
        batch_size = hp['batch_size']
        dropout = hp['dropout']
        learning_rate = hp['learning_rate']

        print(f'[TRAINING START] {hostname}: n_steps={n_steps}, steps_ahead={steps_ahead}, units={lstm_units}, epochs={epochs}, batch={batch_size}, dropout={dropout}, lr={learning_rate}')

        # Get data workload
        df = workload.get(hostname)

        # Use steps_ahead for training target
        X, y, df_arr = concatenate_samples(df, n_steps, steps_ahead)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        # Split train and validation (80/20)
        split_index = len(X) * 8 // 10
        Xtrain, Xval = X[:split_index], X[split_index:]
        ytrain, yval = y[:split_index], y[split_index:]

        # Define model with dropout
        model = Sequential()
        model.add(LSTM(lstm_units, input_shape=(n_steps, n_features)))
        model.add(Dropout(dropout))
        model.add(Dense(1))

        # Custom optimizer with learning rate
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='mse')

        # Model checkpoint
        filepath = f'models/{hostname}/weights-{epochs:02d}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                     save_best_only=True, save_weights_only=False, mode='auto')
        callbacks_list = [checkpoint]

        # Fit model with validation split
        history = model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size,
                           verbose=0, validation_data=(Xval, yval), callbacks=callbacks_list)

        # Save model named as validation loss value
        val_loss = history.history['val_loss'][-1]
        val_loss = '.'.join([str(val_loss).split('.')[0].zfill(3),
                            str(val_loss).split('.')[-1][:6]])  # Keep 6 decimal places
        time_stamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

        model.load_weights(filepath)
        # Include steps_ahead in filename
        filename = f'{val_loss}_{time_stamp}_{epochs}_{n_steps}_{lstm_units}_{steps_ahead}ahead.keras'
        model.save(f'models/{hostname}/{filename}')

        print(f'[MODEL SAVED] {hostname}: {filename} (val_loss: {val_loss}, steps_ahead: {steps_ahead})')

        return model
    except Exception as e:
        print(f'[TRAINING ERROR] {hostname}: Unable to train now: {e}')

def select_best_model(hostname):
    """Select best model based on validation loss value."""
    model_dir = f'./models/{hostname}'
    models = [f for f in os.listdir(model_dir) if f.endswith('.keras')]

    if not models:
        return None

    # Parse validation loss from filenames (format: {val_loss}_{timestamp}_{epochs}_{n_steps}_{units}_{steps_ahead}ahead.keras)
    model_losses = []
    for model_file in models:
        try:
            # Try new format with steps_ahead
            parts = model_file.replace('.keras', '').split('_')
            if len(parts) >= 4:
                loss = float(parts[0])
            else:
                # Fallback to old format: {loss}_{timestamp}_{epochs}.keras
                loss = float(parts[0])
            model_losses.append((loss, model_file))
        except (ValueError, IndexError):
            continue

    if not model_losses:
        return None

    # Sort by validation loss and return best
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
                model, loss, filename, timestamp = self.model_cache[hostname]
                if (datetime.now() - timestamp).total_seconds() < 600:
                    print(f'[CACHE HIT] {hostname}: Using cached model {filename} (loss: {loss})')
                    return model
                else:
                    print(f'[CACHE EXPIRED] {hostname}: Model {filename} expired after {int((datetime.now() - timestamp).total_seconds())}s')
                    del self.model_cache[hostname]

        # Try loading from disk
        try:
            best_model_file = select_best_model(hostname)
            if best_model_file:
                model = load_model(f'models/{hostname}/{best_model_file}')
                loss = float(best_model_file.split('_')[0])

                with self.cache_lock:
                    self.model_cache[hostname] = (model, loss, best_model_file, datetime.now())

                print(f'[CACHE MISS] {hostname}: Loaded model {best_model_file} from disk (loss: {loss})')
                return model
            else:
                print(f'[CACHE MISS] {hostname}: No model available on disk')
        except Exception as e:
            print(f'[CACHE ERROR] {hostname}: Failed to load model: {e}')

        return None

    def get_best_model_with_filename(self, hostname):
        """Get best trained model with filename without blocking."""
        # Check cache first
        with self.cache_lock:
            if hostname in self.model_cache:
                model, loss, filename, timestamp = self.model_cache[hostname]
                if (datetime.now() - timestamp).total_seconds() < 600:
                    return model, filename
                else:
                    del self.model_cache[hostname]

        # Try loading from disk
        try:
            best_model_file = select_best_model(hostname)
            if best_model_file:
                model = load_model(f'models/{hostname}/{best_model_file}')
                loss = float(best_model_file.split('_')[0])

                with self.cache_lock:
                    self.model_cache[hostname] = (model, loss, best_model_file, datetime.now())

                return model, best_model_file
        except Exception:
            pass

        return None, None

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
                    model = train_lstm_model(hostname, steps_ahead=6)

                    if model:
                        # Validate prediction quality
                        # Get n_steps from best model filename, or default to 30
                        try:
                            best_file = select_best_model(hostname)
                            if best_file:
                                parts = best_file.split('_')
                                n_steps = int(parts[3]) if len(parts) >= 4 else 30
                            else:
                                n_steps = 30
                        except:
                            n_steps = 30
                        x_input = array(df[-n_steps:].mem.values)
                        x_input = x_input.reshape((1, n_steps, 1))
                        predict = model.predict(x_input, verbose=0)[0][0]
                        actual = ram_usage.get(hostname)

                        error = abs(predict - actual)

                        if error < 5:
                            best_file = select_best_model(hostname)
                            if best_file:
                                # Parse current loss from filename
                                current_loss = float(best_file.split('_')[0])

                                # Calculate new weighted loss (combine old with new error)
                                # This smooths the loss over multiple predictions
                                new_loss = (current_loss * 0.7) + (error * 0.3)

                                # Update model filename with new loss
                                new_filename = update_model_loss(hostname, best_file, new_loss)

                                # Update cache
                                loss = float(best_file.split('_')[0])
                                filename = new_filename if new_filename else best_file
                                with self.cache_lock:
                                    self.model_cache[hostname] = (model, loss, filename, datetime.now())

                        print(f'[PREDICTION] {hostname}: predicted={predict:.2f}, actual={actual:.2f}, error={error:.2f}')

                sleep(30)  # Rate limit: max 2 trains/minute/host

            except Exception as e:
                print(f'[TRAINING ERROR] {hostname}: {e}')
                sleep(30)


def lstm(hostname, steps_ahead=6):
    """
    Non-blocking LSTM prediction using best available model.
    Never blocks on training operations - uses cached model or falls back gracefully.

    Args:
        hostname: Hostname to predict RAM for
        steps_ahead: Number of steps ahead to predict (default: 6 for 3 minutes)

    Returns:
        Predicted RAM value or current RAM as fallback
    """
    # Check if a trained model is available
    best_model, model_file = lstm_manager.get_best_model_with_filename(hostname)

    if best_model:
        try:
            df = workload.get(hostname)

            # Get n_steps and loss from model file
            if model_file:
                parts = model_file.split('_')
                n_steps = int(parts[3]) if len(parts) >= 4 else 30
                loss = float(parts[0]) if len(parts) >= 1 else 0
            else:
                n_steps = 30
                loss = 0
                model_file = "unknown"

            print(f'[PREDICTION START] {hostname}: Using model {model_file} (loss: {loss:.6f}, steps_ahead: {steps_ahead})')

            # Prepare input data (need at least n_steps samples)
            if len(df) >= n_steps:
                n_features = 1
                x_input = array(df[-n_steps:].mem.values)
                x_input = x_input.reshape((1, n_steps, n_features))

                # Make prediction (direct, not iterative)
                predict = best_model.predict(x_input, verbose=0)[0][0]

                # Validation: Check if prediction is coherent with last value
                if len(df) > 0:
                    last_value = df.iloc[-1]['mem']
                    diff = abs(predict - last_value)

                    if diff > 20 and diff > abs(predict) * 0.20:
                        # Prediction too different from last value, use current RAM
                        actual_ram = ram_usage.get(hostname)
                        print(f'[PREDICTION INCOHERENT] {hostname}: LSTM={predict:.2f}, Last={last_value:.2f}, Diff={diff:.2f}, Using RAM: {actual_ram:.2f}')
                        return actual_ram

                print(f'LSTM prediction for {hostname}: {predict:.2f} (model: {model_file}, steps_ahead: {steps_ahead})')
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
