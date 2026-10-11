"""Cooperative cancellation for CPU HTTP clients. Never signals a GPU worker."""
import json
import io
import select
import socket
import ssl
import threading
import time
from pathlib import Path


class Cancelled(RuntimeError):
    pass


class _PollingReader(io.RawIOBase):
    """Do not depend on shutdown() interrupting a Windows blocking socket read."""
    def __init__(self, sock, control, deadline):
        self.sock, self.control, self.deadline = sock, control, deadline
        self.raw = sock.makefile('rb', buffering=0)  # retains fd across HTTP/1.0 connection close

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            self.control.check()
            if time.monotonic() >= self.deadline:
                raise TimeoutError('request deadline exceeded')
            try:
                value = self.raw.readinto(buffer)
                if value is not None:
                    return value
            except ssl.SSLWantWriteError:
                select.select([], [self.sock], [], .05)
                continue
            except ssl.SSLWantReadError:
                pass
            select.select([self.sock], [], [], .05)

    def close(self):
        self.raw.close()
        super().close()


class ResponseSocket:
    """Use after HTTP request upload, before getresponse; reads poll cancellation and deadline."""
    def __init__(self, sock, control, deadline):
        self.sock, self.control, self.deadline = sock, control, deadline
        sock.setblocking(False)

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def makefile(self, mode):
        if mode != 'rb':
            raise ValueError('response socket supports binary reads only')
        return io.BufferedReader(_PollingReader(self.sock, self.control, self.deadline))


class RequestControl:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.sockets = set()
        self.thread = None
        self.requests_complete = True

    def check(self):
        if self.cancelled.is_set() or (self.path and self.path.exists()):
            self.cancel()
            raise Cancelled('benchmark cancellation requested')

    def register(self, sock):
        with self.lock:
            self.sockets.add(sock)
        self.check()

    def unregister(self, sock):
        with self.lock:
            self.sockets.discard(sock)

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for sock in self.sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def monitor(self):
        while not self.done.wait(.05):
            if self.path.exists():
                self.cancel()
                return

    def __enter__(self):
        if self.path:
            self.thread = threading.Thread(target=self.monitor, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        if self.thread:
            self.thread.join()
        with self.lock:
            active = len(self.sockets)
        if active:
            raise RuntimeError('client sockets still active; cancellation cannot be acknowledged')
        if self.path:
            # This acknowledges CLIENT exit, not a GPU/engine abort. Server idle is checked separately.
            ack = self.path.with_suffix(self.path.suffix + '.ack')
            tmp = ack.with_suffix(ack.suffix + '.tmp')
            tmp.write_text(json.dumps({'client_closed': True, 'cancelled': self.cancelled.is_set(),
                                       'requests_complete': self.requests_complete}), encoding='utf-8')
            tmp.replace(ack)
