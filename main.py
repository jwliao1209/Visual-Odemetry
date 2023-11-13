import os
import glob
import argparse
import numpy as np

from src.vo import VisualOdemetry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="frames",
                        help="directory of sequential frames")
    parser.add_argument("--camera_parameters", type=str, default="camera_parameters.npy",
                        help="npy file of camera parameters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
    frame_paths = sorted(list(glob.glob(os.path.join(args.data, "*.png"))))
    vo = VisualOdemetry(K=camera_params["K"], dist=camera_params["dist"], frame_paths=frame_paths)
    vo.run()
