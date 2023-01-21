
from pathlib import Path
import numpy as np

def cached_array(source_name, names=[], data_dir_trait='data_dir'):
    """Load the contents of an array with the specified name
    if it exists and is newer than source
    """

    def access_cached_array(function):

        def wrapped_call(self):
            source_file = Path(getattr(self, source_name))
            data_dir = getattr(self, data_dir_trait)
            cache_dir = Path(data_dir) / "cache"
            if isinstance(names, str):
                names_ = [names]
            else:
                names_ = names
            all_names = "_".join(names_)
            cached_attr_name = '_cached_' + all_names
            if hasattr(self, cached_attr_name):
                return getattr(self, cached_attr_name)
            cache_file = cache_dir / "cached_{}.npz".format(all_names)
            if cache_dir.is_dir():
                if cache_file.exists():
                    cache_ctime = cache_file.stat().st_ctime
                    source_ctime = source_file.stat().st_ctime
                    if cache_ctime > source_ctime:
                        loaded = np.load(cache_file)
                        if isinstance(names, str):
                            return loaded[names]
                        else:
                            return [loaded[name] for name in names_]
            else:
                cache_dir.mkdir()
            return_value = function(self)
            setattr(self, cached_attr_name, return_value)
            if isinstance(names, str):
                all_arrays = [return_value]
            else:
                all_arrays = return_value
            arr_dict = {name: array for
                        name, array in zip(names_, all_arrays)}
            np.savez_compressed(cache_file, **arr_dict)
            return return_value

        return wrapped_call

    return access_cached_array
