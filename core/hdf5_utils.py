from contextlib import contextmanager
import h5py


@contextmanager
def hdf_open(filename, access="r"):
    hdf = h5py.File(filename, access)
    yield hdf
    hdf.close()


def get_or_add_group(hdf_file, group_name):
    try:
        return hdf_file[group_name]
    except KeyError:
        return hdf_file.create_group(group_name)


def write_weather_to_hdf(output_hdf, cultivars_weather_data):
    for cultivar, weather_data in cultivars_weather_data.items():
        cultivar_group = get_or_add_group(output_hdf, cultivar)
        for name, data in weather_data.items():
            cultivar_group.create_dataset(
                name,
                shape=data.shape,
                dtype=data.dtype,
                data=data, compression="gzip")
