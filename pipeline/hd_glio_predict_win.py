"""
hd_glio_predict_win.py - `hd_glio_predict` with nnU-Net's worker processes
replaced by threads.

nnU-Net v1 hands `trainer.preprocess_patient` (a bound method -> the whole
trainer incl. the CUDA network and a lambda softmax) to a
multiprocessing.Process. That only works with fork (Linux); on Windows'
spawn it dies with "Can't pickle <lambda> ... nd_softmax". For a single
case threads cost nothing, so swap them in and call the stock CLI.

Same arguments as hd_glio_predict (-t1 -t1c -t2 -flair -o).
"""

import queue
import threading
from multiprocessing.pool import ThreadPool


class _ThreadProcess(threading.Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args, daemon=True)

    def terminate(self):  # predict.py calls this on still-alive workers
        pass


class _ThreadQueue(queue.Queue):
    def close(self):  # multiprocessing.Queue API, called when the generator ends
        pass


if __name__ == "__main__":
    import nnunet.inference.predict as predict
    predict.Process = _ThreadProcess
    predict.Queue = _ThreadQueue
    predict.Pool = ThreadPool

    from hd_glio.hd_glio_predict import main
    main()
