'''
this file is modified from keras implemention of data process multi-threading,
see https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
'''
import time
import os
import numpy as np
import threading
import multiprocessing
try:
    import queue
except ImportError:
    import Queue as queue


class GeneratorEnqueuer():
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        if os.name is 'nt' and use_multiprocessing is True:
            # On Windows, avoid **SYSTEMATIC** error in `multiprocessing`:
            # `TypeError: can't pickle generator objects`
            # => Suggest multithreading instead of multiprocessing on Windows
            raise ValueError('Using a generator with `use_multiprocessing=True`'
                             ' is not supported on Windows (no marshalling of'
                             ' generators across process boundaries). Instead,'
                             ' use single thread/process or multithreading.')
        else:
            self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self._manager = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            if self._use_multiprocessing is False:
                while not self._stop_event.is_set():
                    with self.genlock:
                        try:
                            if self.queue is not None and self.queue.qsize() < self.max_queue_size:
                                # On all OSes, avoid **SYSTEMATIC** error in multithreading mode:
                                # `ValueError: generator already executing`
                                # => Serialize calls to infinite iterator/generator's next() function
                                generator_output = next(self._generator)
                                self.queue.put(generator_output)
                            else:
                                time.sleep(self.wait_time)
                        except StopIteration:
                            break
                        except Exception as e:
                            # Can't pickle tracebacks.
                            # As a compromise, print the traceback and pickle None instead.
                            if not hasattr(e, '__traceback__'):
                                setattr(e, '__traceback__', sys.exc_info()[2])
                            self.queue.put((False, e))
                            self._stop_event.set()
                            break
            else:
                while not self._stop_event.is_set():
                    try:
                        if self.queue is not None and self.queue.qsize() < self.max_queue_size:
                            generator_output = next(self._generator)
                            self.queue.put(generator_output)
                        else:
                            time.sleep(self.wait_time)
                    except StopIteration:
                        break
                    except Exception as e:
                        # Can't pickle tracebacks.
                        # As a compromise, print the traceback and pickle None instead.
                        traceback.print_exc()
                        setattr(e, '__traceback__', None)
                        self.queue.put((False, e))
                        self._stop_event.set()
                        break
        try:
            self.max_queue_size = max_queue_size
            if self._use_multiprocessing:
                self._manager = multiprocessing.Manager()
                self.queue = self._manager.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                # On all OSes, avoid **SYSTEMATIC** error in multithreading mode:
                # `ValueError: generator already executing`
                # => Serialize calls to infinite iterator/generator's next() function
                self.genlock = threading.Lock()
                self.queue = queue.Queue(maxsize=max_queue_size)
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.random_seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called `start()`.
        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if self._use_multiprocessing:
                if thread.is_alive():
                    thread.terminate()
            else:
                # The thread.is_alive() test is subject to a race condition:
                # the thread could terminate right after the test and before the
                # join, rendering this test meaningless -> Call thread.join()
                # always, which is ok no matter what the status of the thread.
                thread.join(timeout)

        if self._manager:
            self._manager.shutdown()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)
