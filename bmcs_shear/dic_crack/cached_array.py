
from pathlib import Path
import numpy as np

def cached_array(source_name, names=[], data_dir_trait='data_dir', refresh=False):
    """Load the contents of an array with the specified name if it exists and is newer than source.

    Args:
        source_name (str): The name of the source file.
        names (list): List of names of the arrays to load.
        data_dir_trait (str): The name of the data directory trait.
        refresh (bool): Force the refresh of a cached array

    Returns:
        function: Wrapper function that checks if force_array_refresh is True to ignore cached value.

    Examples:
        @cached_array(source_name="beam_param_file", names=['tstring', 'time', 'F'], data_dir_trait='data_dir')
        def access_cached_array(self):
            # Function implementation
    """

    def access_cached_array(function):

        def wrapped_call(self):
            force_array_refresh = getattr(self, 'force_array_refresh', False) or refresh
            if force_array_refresh:
                print('forced refreshing of cached arrays for', self)

            source_file = Path(getattr(self, source_name))
            data_dir = getattr(self, data_dir_trait)
            cache_dir = Path(data_dir) / "cache"
            names_ = [names] if isinstance(names, str) else names
            cached_attr_name = f'_cached_{"_".join(names_)}'
            cache_file = cache_dir / f"cached_{'_'.join(names_)}.npz"
            if force_array_refresh and cache_file.exists():
                cache_file.unlink()
            if hasattr(self, cached_attr_name):
                return getattr(self, cached_attr_name)
            if cache_dir.is_dir():
                if cache_file.exists():
                    cache_ctime = cache_file.stat().st_ctime
                    source_ctime = source_file.stat().st_ctime
                    if cache_ctime > source_ctime:
                        loaded = np.load(cache_file)
                        return loaded[names] if isinstance(names, str) else [loaded[name] for name in names_]
            else:
                cache_dir.mkdir()
            return_value = function(self)
            setattr(self, cached_attr_name, return_value)
            all_arrays = [return_value] if isinstance(names, str) else return_value
            np.savez_compressed(cache_file, **dict(zip(names_, all_arrays)))
            return return_value

        return wrapped_call

    return access_cached_array

