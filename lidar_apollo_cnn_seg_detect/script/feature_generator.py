from open3d import *
import numpy as np

class Feature_generator():
    def __init__(self, pc_f):
        pc = self.load_pc_from_file(pc_f)
        self.pc = pc
        self.range = 60
        self.width = 512
        self.height = 512
        self.min_height = -5.0
        self.max_height = 5.0
        self.feature = np.zeros((self.height, self.width, 8))

        for row in range(self.height):
            for col in range(self.width):
                idx = row * self.width + col
                center_x = self.Pixel2pc(row, self.height, self.range)
                center_y = self.Pixel2pc(col, self.width, self.range)
                # direction
                self.feature[row, col, 0] \
                    = np.arctan2(center_x, center_y) / (2.0 * np.pi)
                # = np.arctan2(center_y, center_x) * 180 / np.pi
                # distance
                self.feature[row, col, 1] \
                    = np.hypot(center_x, center_y) / 60 - 0.5

    def load_pc_from_file(self, pc_f):
        return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 4])

    def Pc2pixel(self, in_pc, in_range, out_size):
        inv_res = 0.5 * out_size / in_range
        return int((in_range - in_pc) * inv_res)

    def Pixel2pc(self, in_pixel, in_size, out_range):
        res = 2.0 * out_range / in_size
        return out_range - in_pixel * res

if __name__ == "__main__":
    feature_generator = Feature_generator("../pcd/sample.pcd.bin")
    # print(feature_generator.pc)
    # print(feature_generator.feature[0, 0, 0])
    # print(feature_generator.feature[511, 0, 0])
    # print(feature_generator.feature[0, 511, 0])
    # print(feature_generator.feature[511, 511, 0])

    # print(feature_generator.feature[255, 0, 0])
    # print(feature_generator.feature[0, 0, 1])
    # print(feature_generator.feature[0, 255, 1])
    # print(feature_generator.feature[255, 255, 1])
    # print(feature_generator.feature[511, 0, 1])
