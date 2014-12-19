from __future__ import division

## Kyle Dickerson
## kyle.dickerson@gmail.com
## Jan 15, 2008
##
## Self-organizing map using scipy
## This code is licensed and released under the GNU GPL

import random
import math
import sys
import scipy
import scipy.ndimage
from PIL import Image


# returns the row, column pair that corresponds to the 1-d index of a 2-d array with given width
def find_indices(loc, width):
    r = 0
    while loc >= width:
        loc -= width
        r += 1
    c = loc
    return (r, c)

class SOM:
    # FV_ranges allows you to specify the range of each feature in the feature vector
    # This lets you workaround the problem of having all of your features outside the range of the initialization values
    # Which, from my tests, may result in bad results
    def __init__(self, height=10, width=10, FV_size=10, learning_rate=0.1, FV_ranges=None):
        self.height = height
        self.width = width
        self.FV_size = FV_size
        #self.nodes = scipy.random.random((width, height, FV_size))
        if not FV_ranges:
            self.nodes = scipy.random.uniform(0,100,(width, height, FV_size))
        elif len(FV_ranges) == 1:
            self.nodes = scipy.random.uniform(FV_ranges[0][0],FV_ranges[0][1],(width, height, FV_size))
        else:
            self.nodes = scipy.array( [[[random.uniform(FV_ranges[i][0], FV_ranges[i][1]) for i in range(FV_size)] for j in range(width)] for k in range(height)])
        
        self.learning_rate = learning_rate
        self.radius = (height+width)/4

    # train_vector: [ FV0, FV1, FV2, ...] -> [ [...], [...], [...], ...]
    def train(self, iterations, train_vector, iterative_update=False, grow=True, num_pts_to_grow=3):
        self.iterations = iterations
        
        for t in range(len(train_vector)):
            train_vector[t] = scipy.array(train_vector[t])
        delta_nodes = scipy.zeros((self.width, self.height, self.FV_size), float)
        
        for i in range(0, iterations):
            cur_radius = self.radius_decay(i)
            cur_lr = self.learning_rate_decay(i)
            sys.stdout.write("\rTraining Iteration: " + str(i+1) + "/" + str(iterations))
            sys.stdout.flush()
            
            # Grow the map where it's doing worst
            if grow and not (i % 20):
                for iik in range(num_pts_to_grow):
                    dist_mask = self.build_distance_mask()
                    worst_loc = find_indices(scipy.argmax(dist_mask), self.width)
                    worst_row = worst_loc[0]
                    worst_col = worst_loc[1]
                    # Insert the row
                    prev_row = worst_row - 1 if worst_row-1 >= 0 else self.height - 1
                    next_row = worst_row + 1 if worst_row+1 < self.height else 0
                    self.nodes = scipy.insert(self.nodes, worst_row, [[0]], axis=0)
                    self.height += 1
                    # Fill the new row with interpolated values
                    for col in range(self.width):
                        self.nodes[worst_row, col] = (self.nodes[prev_row, col] + self.nodes[next_row, col]) / 2
                    # Insert the column
                    prev_col = worst_col - 1 if worst_col-1 >= 0 else self.width - 1
                    next_col = worst_col + 1 if worst_col+1 < self.width else 0
                    self.nodes = scipy.insert(self.nodes, worst_col, [[0]], axis=1)
                    self.width += 1
                    # Fill the new column with interpolated values
                    for row in range(self.height):
                        self.nodes[row, worst_col] = (self.nodes[row, prev_col] + self.nodes[row, next_col]) / 2
                self.radius = (self.height+self.width)/4
                delta_nodes = scipy.zeros((self.width, self.height, self.FV_size), float)
            
            if not iterative_update:
                delta_nodes.fill(0)
            else:
                random.shuffle(train_vector)
            
            for j in range(len(train_vector)):
                best = self.best_match(train_vector[j])
                # pick out the nodes that are within our decaying radius:
                for loc in self.find_neighborhood(best, cur_radius):
                    influence = (-loc[2] + cur_radius) / cur_radius  # linear scaling of influence
                    inf_lrd = influence*cur_lr
                    delta_nodes[loc[0],loc[1]] += inf_lrd*(train_vector[j]-self.nodes[loc[0],loc[1]])
                if iterative_update:
                    self.nodes += delta_nodes
                    delta_nodes.fill(0)
            if not iterative_update:
		delta_nodes /= len(train_vector)
                self.nodes += delta_nodes
        sys.stdout.write("\n")

    def smooth(self):
        self.nodes = scipy.ndimage.gaussian_filter(self.nodes, 0.5)
    
    def radius_decay(self, itr):
        return ((self.iterations - itr) / self.iterations) * self.radius

    # Update the learning rate
    def learning_rate_decay(self, itr):
        return ((self.iterations - itr) / self.iterations) * self.learning_rate
    
    # pt is (row, column)
    def find_neighborhood(self, pt, dist):
        # returns a chessboard distance neighborhood, with distances determined by Euclidean distance
        #   - Meaning, take a square around the center pt
        dist = int(dist)
        # This locks the grid at the edges
        #min_y = max(int(pt[0] - dist), 0)
        #max_y = min(int(pt[0] + dist)+1, self.height)
        #min_x = max(int(pt[1] - dist), 0)
        #max_x = min(int(pt[1] + dist)+1, self.width)
        
        # This allows the grid to wrap vertically and horizontally
        min_y = int(pt[0] - dist)
        max_y = int(pt[0] + dist)+1
        min_x = int(pt[1] - dist)
        max_x = int(pt[1] + dist)+1
        
        # just build the cross product of the bounds
        neighbors = []
        for y in range(min_y, max_y):
            y_piece = (y-pt[0])**2
            y = y + self.height if y < 0 else y % self.height
            for x in range(min_x, max_x):
                # Manhattan
                # d = abs(y-pt[0]) + abs(x-pt[1])
                # Euclidean
                d = (y_piece + (x-pt[1])**2)**0.5
                x = x + self.width  if x < 0 else x % self.width
                neighbors.append((y,x,d))
        return neighbors
    
    # Returns location of best match
    # target_FV is a scipy array
    def best_match(self, target_FV):
        # Euclidean distance, computed over the entire net, hopefully this is faster
        loc = scipy.argmin((((self.nodes - target_FV)**2).sum(axis=2))**0.5)
        return find_indices(loc, self.width)

    # returns the distance between two Feature Vectors
    # FV_1, FV_2 are scipy arrays
    def FV_distance(self, FV_1, FV_2):
        # Euclidean distance
        dist = (sum((FV_1 - FV_2)**2))**0.5
        return dist
    
    def build_distance_mask(self):
        tmp_nodes = scipy.zeros((self.width, self.height), float)
        for r in range(self.height):
            for c in range(self.width):
                for n in self.find_neighborhood((r,c), 1):
                    tmp_nodes[r,c] += self.FV_distance(self.nodes[r,c], self.nodes[n[0],n[1]])
        return tmp_nodes
    
    # Show smoothness of the SOM.  The darker the area the more rapid the change, generally bad.
    def save_similarity_mask(self, filename, path="."):
        tmp_nodes = self.build_distance_mask()
        #tmp_nodes -= tmp_nodes.min()
        tmp_nodes *= 255 / tmp_nodes.max()
        tmp_nodes = 255 - tmp_nodes
        img = Image.new("L", (self.width, self.height))
        for r in range(self.height):
            for c in range(self.width):
                img.putpixel((c,r), tmp_nodes[r,c])
        img = img.resize((self.width*10,self.height*10),Image.NEAREST)
        img.save(path + "/" + filename + ".png")
        
###### 
    def save_colors(self, iter):
        img = Image.new("RGB", (self.width, self.height))
        for r in range(self.height):
            for c in range(self.width):
                img.putpixel((c,r), (int(self.nodes[r,c,0]), int(self.nodes[r,c,1]), int(self.nodes[r,c,2])))
        img = img.resize((self.width*10,self.height*10),Image.NEAREST)
        img.save("som_color_"+str(iter)+".png")
###### 

if __name__ == "__main__":
    print "Initialization..."
    colors = [[0, 0, 0], [255, 255, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [0, 0, 255]]
    color_som = SOM(32,32,3,0.1,[(0,255)])
    print "Training for colors function..."
    color_som.train(200, colors, False)
    color_som.save_colors("test")
    color_som.save_similarity_mask("test_sim")
    
