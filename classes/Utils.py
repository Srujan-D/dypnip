import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class Utils:
    def isWall(self, obs):
        x = [item[0] for item in obs.allCords]
        y = [item[1] for item in obs.allCords]
        if(len(np.unique(x)) < 2 or len(np.unique(y)) < 2):
            return True  # Wall
        else:
            return False  # Rectangle

    def drawMap(self, obs, curr, dest):
        fig = plt.figure()
        currentAxis = plt.gca()
        for ob in obs:
            if(self.isWall(ob)):
                x = [item[0] for item in ob.allCords]
                y = [item[1] for item in ob.allCords]
                plt.scatter(x, y, c="red")
                plt.plot(x, y)
            else:
                currentAxis.add_patch(Rectangle(
                    (ob.bottomLeft[0], ob.bottomLeft[1]), ob.width, ob.height, alpha=0.4))

        plt.scatter(curr[:,0], curr[:,1], s=25, c='green', zorder=20)
        plt.scatter(dest[:,0], dest[:,1], s=25, c='red', zorder=20)
        fig.canvas.draw()

    # source: https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    @staticmethod
    def get_row_compressor(old_dimension, new_dimension):
        dim_compressor = np.zeros((new_dimension, old_dimension))
        bin_size = float(old_dimension) / new_dimension
        next_bin_break = bin_size
        which_row = 0
        which_column = 0
        while which_row < dim_compressor.shape[0] and which_column < dim_compressor.shape[1]:
            if round(next_bin_break - which_column, 10) >= 1:
                dim_compressor[which_row, which_column] = 1
                which_column += 1
            elif next_bin_break == which_column:

                which_row += 1
                next_bin_break += bin_size
            else:
                partial_credit = next_bin_break - which_column
                dim_compressor[which_row, which_column] = partial_credit
                which_row += 1
                dim_compressor[which_row, which_column] = 1 - partial_credit
                which_column += 1
                next_bin_break += bin_size
        dim_compressor /= bin_size
        return dim_compressor

    @staticmethod
    def get_column_compressor(old_dimension, new_dimension):
        return Utils.get_row_compressor(old_dimension, new_dimension).transpose()
    
    @staticmethod
    def compress_and_average(array, new_shape):
        # Note: new shape should be smaller in both dimensions than old shape
        new_mat = np.mat(Utils.get_row_compressor(array.shape[0], new_shape[0])) * \
            np.mat(array) * \
            np.mat(Utils.get_column_compressor(array.shape[1], new_shape[1]))
        return np.squeeze(np.asarray(new_mat))