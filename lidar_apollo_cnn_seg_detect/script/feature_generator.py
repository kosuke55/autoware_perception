from open3d import *
import numpy as np
import baidu_cnn_util as bcu


class Feature_generator():
    def __init__(self):
        self.range = 60
        self.width = 512
        self.height = 512
        self.siz = self.width * self.height
        self.min_height = -5.0
        self.max_height = 5.0

        self.log_table = np.zeros(256)
        for i in range(len(self.log_table)):
            self.log_table[i] = np.log1p(i)
        # print(self.log_table)

        self.max_height_data = 0
        self.mean_height_data = 1
        self.count_data = 2
        self.direction_data = 3
        self.top_intensity_data = 4
        self.mean_intensity_data = 5
        self.distance_data = 6
        self.nonempty_data = 7
        self.feature = np.zeros((self.siz, 8))

        for row in range(self.height):
            for col in range(self.width):
                idx = row * self.width + col
                center_x = self.Pixel2pc(row, self.height, self.range)
                center_y = self.Pixel2pc(col, self.width, self.range)
                # direction
                self.feature[idx, self.direction_data] \
                    = np.arctan2(center_x, center_y) / (2.0 * np.pi)
                # = np.arctan2(center_y, center_x) * 180 / np.pi (for debug)
                # distance
                self.feature[idx, self.distance_data] \
                    = np.hypot(center_x, center_y) / 60 - 0.5

    def logCount(self, count):
        if(count < len(self.log_table)):
            return self.log_table[count]
        else:
            return np.log(1 + count)

    def load_pc_from_file(self, pc_f):
        return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 4])

    def Pc2pixel(self, in_pc, in_range, out_size):
        inv_res = 0.5 * out_size / in_range
        return int((in_range - in_pc) * inv_res)

    def Pixel2pc(self, in_pixel, in_size, out_range):
        res = 2.0 * out_range / in_size
        return out_range - in_pixel * res

    def generate(self, pc_f):
        points = self.load_pc_from_file(pc_f)
        print("points.shape = " + str(points.shape))
        self.map_idx = np.zeros(len(points))
        inv_res_x = 0.5 * self.width / self.range
        inv_res_y = 0.5 * self.height / self.range


        # for i in range(len(points)):
        for i, point in enumerate(points):
            if(point[2] <= self.min_height or
               point[2] <= self.max_height):
                self.map_idx[i] = -1
            # project point cloud to 2d map. clac in which grid point is.
            # * the coordinates of x and y are exchanged here
            # (row <-> x, column <-> y)
            pos_x = bcu.F2I(point[0], self.range, inv_res_x)
            pos_y = bcu.F2I(point[1], self.range, inv_res_y)
            if(pos_x >= self.width or pos_x < 0 or
               pos_y >= self.height or pos_y < 0):
                self.map_idx[i] = -1
                continue
            self.map_idx[i] = pos_y * self.width + pos_x
            idx = int(self.map_idx[i])
            pz = point[2]
            # if use nuscenes divide intensity by 255.
            pi = point[3] / 255.0
            if(self.feature[idx, self.max_height_data] < pz):
                self.feature[idx, self.max_height_data] = pz
                # not I_max but I of z_max ?
                self.feature[idx, self.top_intensity_data] = pi

            self.feature[idx, self.mean_height_data] += pz
            self.feature[idx, self.mean_intensity_data] += pi
            self.feature[idx, self.count_data] += 1.0

        for i in range(self.siz):
            eps = 1e-6
            if(self.feature[i, self.count_data] < eps):
                self.feature[i, self.max_height_data] = 0.0
            else:
                self.feature[i, self.mean_height_data] \
                    /= self.feature[i, self.count_data]
                self.feature[i, self.mean_intensity_data] \
                    /= self.feature[i, self.count_data]
                self.feature[i, self.nonempty_data] = 1.0
            self.feature[i, self.count_data] \
                = self.logCount(int(self.feature[i, self.count_data]))

if __name__ == "__main__":
    import time
    start = time.time()
    feature_generator = Feature_generator()
    feature_generator.generate("../pcd/sample.pcd.bin")
    end = time.time()
    print(end - start)
    # print(feature_generator.feature)
