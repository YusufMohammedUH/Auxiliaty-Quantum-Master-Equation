# %%
import h5py
from h5py._hl.files import File
from typing import Dict, Union, List
import numpy as np


def create_hdf5(filename: str, data: Union[np.ndarray, List] = None) -> None:
    """
    Create HDF5 file and save data in it if data is not None.

    Parameters
    ----------
    filename : str
        Name of HDF5 file.

    data : Union[np.ndarray, List], optional
        Data to write to HDF5 file.
    """
    with h5py.File(filename, 'w') as f:
        if data is not None:
            f.create_dataset('data', data=data)


def add_attrs(file: Union[str, File], dir: str, attrs: Dict) -> None:
    """Given string 'dir' key, value pairs of dictionary 'attrs' are saved in
    file 'file' as attributes.

    Parameters
    ----------
    file : Union[str, File]
        File name of the HDF5 file or h5py.File object

    dir : str
        Directory to group or dataset to attach attributes

    attrs : Dict
        Dictionary of attributes
    """
    for i in attrs:
        if file is None:
            file[f'{dir}'].attrs[f'{i}'] = attrs[i]
        else:
            with h5py.File(f'{file}', 'a') as f:
                f[f'{dir}'].attrs[f'{i}'] = attrs[i]


def add_data(file: Union[str, File], dir: str, dataname: str, data: np.ndarray
             ) -> None:
    """Given filename file and group dir and dataset dataname, save the data
    to file.

    Parameters
    ----------
    file : Union[str, File]
        File name of the HDF5 file or h5py.File object

    dir : str
        Directory to group to save dataset to

    dataname : str
        Name of dataset

    data : np.ndarray
        Data to be saved
    """
    if type(file) != str:
        file.create_dataset(f'{dir}/{dataname}', data=data)
    else:
        with h5py.File(f'{file}', 'a') as f:
            f.create_dataset(f'{dir}/{dataname}', data=data)


def read_attrs(file: Union[str, File], dir: str) -> Dict:
    """Given filename file and group or dataset dir, return the attributes
    of dir as dictionary

    Parameters
    ----------
    file : Union[str, File]
        File name of the HDF5 file or h5py.File object

    dir : str
        Group or dataset name

    Returns
    -------
    Dict
        Attributes of dir as dictionary
    """
    if type(file) != str:
        return file[f'{dir}'].attrs
    else:
        with h5py.File(file, 'r') as f:
            attrs = dict(f[f'{dir}'].attrs)
        return attrs


def read_data(file: Union[str, File], dir: str, dataname: str) -> np.ndarray:
    """Given filename file and group dir and dataset dataname, return the
    data of dataname as numpy array

    Parameters
    ----------
    file : Union[str, File]
        File name of the HDF5 file or h5py.File object

    dir : str
        Group name

    dataname : str
        Dataset name

    Returns
    -------
    out: np.ndarray
        Dataset dataname in groupe dir as numpy array
    """
    if type(file) != str:
        return file[f'{dir+"/"+dataname}'][:]
    else:
        with h5py.File(file, 'r') as f:
            data = f[f'{dir+"/"+dataname}'][:]
        return data


if __name__ == '__main__':
    create_hdf5('foo.hdf5')
    add_attrs('foo.hdf5', '/', {'test': 'test'})
    add_data('foo.hdf5', '/green', 'test', [1, 2, 3])
    add_data('foo.hdf5', '/', 'test2', [1, 2, 3])
    add_data('foo.hdf5', '/', 'test3', [1, 2, 3])

# %%