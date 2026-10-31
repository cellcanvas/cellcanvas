"""Napari painting buffer synchronization through the explicit HTTP API."""

import numpy as np
from qtpy.QtCore import QObject, Qt, QTimer, Signal, Slot
from superqt.utils import create_worker


def save_changes(client, context, previous, current, uncertain=None, batch_size=65536):
    """Send immutable voxel assignments in bounded requests, including erases."""
    if current.shape != previous.shape or current.ndim != 3:
        raise ValueError("Painting must keep its original ZYX shape.")
    if not np.issubdtype(current.dtype, np.integer) or np.any(current < 0):
        raise ValueError("Painting labels must be nonnegative integers.")
    old, new = previous.ravel(), current.ravel()
    for start in range(0, new.size, batch_size):
        stop = min(start + batch_size, new.size)
        mask = new[start:stop] != old[start:stop]
        if uncertain is not None:
            mask |= uncertain.ravel()[start:stop]
        changed = np.flatnonzero(mask) + start
        for label in np.unique(new[changed]):
            indices = changed[new[changed] == label]
            coordinates = np.column_stack(np.unravel_index(indices, current.shape)).tolist()
            response = client.send_request("api/painting/update", dict(
                context, coordinates=coordinates, label=int(label)), "POST")
            if (not isinstance(response, dict) or response.get("status") != "success"
                    or response.get("updates") != len(coordinates) or response.get("errors") != 0):
                raise RuntimeError(client.last_error or "The server did not confirm every painted voxel.")
    return current


class PaintingSync(QObject):
    """Keep edits locally until the server confirms them; one request worker at a time.

    This legacy adapter holds the editable volume and a confirmed copy in RAM.
    It assumes one painting writer; it is not a concurrent editing protocol.
    """

    status_changed = Signal(str)

    def __init__(self, layer, client, context, parent=None):
        super().__init__(parent)
        self.layer, self.client, self.context = layer, client, dict(context)
        self.confirmed = np.array(layer.data, copy=True)
        self.uncertain = self._attempt = None
        self.failed = self._dirty = self._closed = False
        self._worker = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.flush)
        self._events = [layer.events.paint, layer.events.labels_update,
                        layer.events.set_data, layer.events.data]
        for emitter in self._events:
            emitter.connect(self.changed)

    @property
    def pending(self):
        return self._dirty or self._worker is not None or self.failed

    def changed(self, event=None):
        if self._closed:
            return
        self._dirty = True
        if not self.failed:
            self.status_changed.emit("Painting has unsaved changes.")
        if self._worker is None and not self.failed:
            self.timer.start()

    @Slot()
    def flush(self):
        if self._closed or self.failed or self._worker is not None or not self._dirty:
            return
        self._dirty = False
        self._attempt = None
        try:
            current = np.array(self.layer.data, copy=True)
            self._attempt = current
            previous, client, context = self.confirmed, self.client, dict(self.context)
            uncertain = self.uncertain
            self.status_changed.emit("Saving painting...")
            worker = create_worker(lambda: save_changes(client, context, previous, current, uncertain),
                                   _start_thread=False, _ignore_errors=True)
            worker.returned.connect(self._saved, Qt.QueuedConnection)
            worker.errored.connect(self._failed, Qt.QueuedConnection)
            worker.finished.connect(self._finished, Qt.QueuedConnection)
            self._worker = worker
            worker.start()
        except Exception as error:
            self._failed(error)
            self._finished()

    @Slot(object)
    def _saved(self, snapshot):
        if not self._closed:
            self.confirmed = snapshot
            self.uncertain = None

    @Slot(object)
    def _failed(self, error):
        if not self._closed:
            # An unsuccessful response can follow a committed or partial write.
            # Re-send these coordinates even if the user has since undone them.
            if self._attempt is not None and self._attempt.shape == self.confirmed.shape:
                changed = self._attempt != self.confirmed
                self.uncertain = changed if self.uncertain is None else self.uncertain | changed
            self.failed = self._dirty = True
            self.status_changed.emit(f"Painting not saved: {error}. Retry to save local edits.")

    @Slot()
    def _finished(self):
        self._worker = None
        self._attempt = None
        if self._closed or self.failed:
            return
        if self._dirty:
            self.timer.start(0)
        else:
            self.status_changed.emit("Painting saved.")

    @Slot()
    def retry(self):
        if not self._closed and self._worker is None:
            self.failed = False
            self._dirty = True
            self.flush()

    def detach(self, *, saved_locally=False):
        """Disconnect after a server save or an explicit local backup.

        A request already in flight may still complete after a local backup.
        """
        if self.pending and not saved_locally:
            raise RuntimeError("Save or retry painting before switching datasets.")
        self._closed = True
        self.timer.stop()
        for emitter in self._events:
            emitter.disconnect(self.changed)
        self.layer.editable = False
