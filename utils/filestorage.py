import pathlib
from sacred.observers import FileStorageObserver

class CustomFileStorageObserver(FileStorageObserver):
    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        if _id is None:
            _id = "baseline"
            (pathlib.Path(self.basedir) / _id).parent.mkdir(exist_ok=True, parents=True)
        return super().started_event(ex_info, command, host_info, start_time, config, meta_info, _id)