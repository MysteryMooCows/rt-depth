import sys
from threading import Thread
import time
from typing import Optional


class ProgressIndicator:
    def __init__(self):
        self._running: bool = False
        self._thread: Optional[Thread] = None
        self._character: str = '.'

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def start(self, character: str = '.') -> None:
        if self._running:
            return
        
        self._running = True
        self._character = character
        self._thread = Thread(target=self._print_progress)
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        
        self._running = False

        if self._thread:
            self._thread.join()

        self._thread = None

    def _print_progress(self) -> None:
        while self._running:
            sys.stdout.write(self._character)
            sys.stdout.flush()
            time.sleep(0.5)