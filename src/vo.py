import cv2
import numpy as np
import open3d as o3d
import multiprocessing as mp

from src.image import ImageFrame
from src.camera import CameraPose
from src.transfrom import expand_vector_dim, reduce_vector_dim


class VisualOdemetry:
    def __init__(self, K, dist, frame_paths):
        self.K = K
        self.dist = dist
        self.frame_paths = frame_paths

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()

        width = 860
        height = 540
        points = np.array(
            [
                [0,       0,        0],
                [0,       0,        1],
                [0,       height-1, 1],
                [width-1, height-1, 1],
                [width-1, 0,        1],
            ]
        )
        points = expand_vector_dim((np.linalg.inv(self.K) @ points.T).T)
        lines = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [0, 3],
                [0, 4],
                [1, 4],
                [2, 4],
                [3, 4]
            ]
        )

        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
            except:
                R, t = None, None

            if R is not None:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector((np.hstack((R, t)) @ points.T).T)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 0], (lines.shape[0],1)))
                vis.add_geometry(line_set)
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()


    def process_frames(self, queue):
        orb = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        camera_pose = CameraPose(R=np.eye(3, dtype=np.float64), t=np.zeros((3, 1), dtype=np.float64))

        prev_frame = ImageFrame(path=self.frame_paths[0], frame=0, K=self.K, dist=self.dist, orb=orb)
        prev_frame.imread()
        prev_frame.undistort()
        prev_frame.detect_features()

        for k, frame_path in enumerate(self.frame_paths[1:]):
            # Step 1: Capture new frame img_{k+1}
            curr_frame = ImageFrame(path=frame_path, frame=k, K=self.K, dist=self.dist, orb=orb)
            curr_frame.imread()
            curr_frame.undistort()

            # Step 2: Extract and match features between img_{k+1} and img_{k} 
            curr_frame.detect_features()
            matches = matcher.match(prev_frame.descriptors, curr_frame.descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            curr_frame.match_key_points(matches, main=True)
            prev_frame.match_key_points(matches, main=False)

            # Step 3: Estimate the essential matrix E_{k, k+1}
            E, mask = cv2.findEssentialMat(prev_frame.matched_points, curr_frame.matched_points, self.K, threshold=0.85, method=cv2.RANSAC)

            # Step 4: Decompose the E_{k, k+1} into R_{k+1}^k and t_{k+1}^k to get the relative pose [R_{k+1}^k t_{k+1}^k]
            _, R_curr, t_curr, mask, X_curr = cv2.recoverPose(
                E, prev_frame.matched_points, curr_frame.matched_points,
                self.K, distanceThresh=50, mask=mask
            )
            X_curr = reduce_vector_dim(X_curr.T)

            # Step 5: Calculate the pose of camera k+1 relative to the first camera
            camera_pose.update(R_curr, t_curr)
            R, t = camera_pose.get()

            queue.put((R, t))
            curr_frame.imshow()

            if cv2.waitKey(30) == 27:
                break

            prev_frame = curr_frame
