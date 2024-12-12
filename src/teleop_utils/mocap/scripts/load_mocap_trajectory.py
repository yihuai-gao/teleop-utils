import numpy as np
import numpy.typing as npt
import pickle
import click

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
def main(data_file: str):
    with open(data_file, "rb") as f:
        data: list[tuple[npt.NDArray[np.float64], float]] = pickle.load(f)
    # print(len(data), data[0])
    print(len(data))
    if len(data) == 0:
        print("No data found")
        return
    pose_xyz_wxyz: npt.NDArray[np.float64] = data[0][0]
    mocap_timestamp: float = data[0][1]
    print(pose_xyz_wxyz, mocap_timestamp)


if __name__ == "__main__":
    main()
