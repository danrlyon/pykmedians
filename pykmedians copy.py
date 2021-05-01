''' Calculate K Medians from CSV values

    TODO:
    - Plot data
    - Ingest csv files
'''



import math
import random
import numpy as np
from sklearn import datasets
import logging
import configargparse
import matplotlib.pyplot as plt


# Constants for Logger
APP_NAME = "kmedians"
DATE_FORMAT = "%Y-%m-%d %H:%M"
MESSAGE_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
log = logging.getLogger(__name__)
def setup_logging(options):
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARN': logging.WARN,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    loglevel = log_levels.get(options.loglevel.upper(), logging.DEBUG)
    log.setLevel(loglevel)
    # STDOUT log handler
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)


def parse_args():
    ''' Parse Command Line arguments '''
    argp = configargparse.ArgParser()
    argp.add(
        '-f',
        '--inputcsv',
        required=False,
        help='The dataset file as a csv to process.'
    )
    argp.add(
        '-k',
        '--kmedians',
        required=False,
        type=int,
        help='The number of k-median centroids.',
        default=2
    )
    argp.add(
        '-d',
        '--dimensions',
        required=False,
        type=int,
        help='The number of dimensions in the data.',
        default=2
    )
    argp.add(
        '-e',
        '--epsilon',
        required=False,
        type=float,
        help='The minimum amount of change in the k-medians to stop iterating.',
        default=0.00001
    )
    argp.add(
        '--loglevel',
        required=False,
        help='Logging Level',
        default='DEBUG',
        env_var='LOG_LEVEL'
    )
    options = argp.parse_args()
    # Debug options
    # print(argp.format_help())
    setup_logging(options)
    log.debug("Parsed command line arguments")
    if options.loglevel.upper() == "DEBUG":
        for line in argp.format_values().split('\n')[0:-1]:
            log.debug(line)
    return options


def initial_centroids(k, dataset):
    ''' The input is k, the number of centroids to generate,
        and dataset, the actual data points to cluster.
        
        This function returns a list of inital k median centroids
        selected by splitting the dataset into k groups and randomly
        picking a data point in each group.
    '''
    dataset_size = len(dataset) 
    log.debug(f"Initializing k={k} Centroids from "
              f"dataset size={dataset_size}.")
    centroids = list()
    group_size = int(len(dataset)/k + 1)
    log.debug(f"The size of each group is {group_size}")
    for i in range(0, dataset_size, group_size):
        log.debug(f"i={i}")
        index = random.randint(i, min(i + group_size, dataset_size - 1)) 
        log.debug(f"Index of centroid: {index}")
        centroids.append(dataset[index])
    log.debug(f"Centroids: {centroids}")
    return centroids


def assign_kmedian_labels(centroids, dataset):
    ''' Return a list the same size as the data set containing the
        index of the centroid for each corresponding item in the
        dataset.    
    '''
    labels = list()
    for i, item in enumerate(dataset):
        labels.append(0)
        distance = math.inf
        for ii in range(len(centroids)):
            new_distance = manhattan_distance(centroids[ii], item)
            if new_distance < distance:
                log.debug(f"Updating label to {ii}")
                distance = new_distance
                labels[i] = ii
    return labels


def manhattan_distance(centroid, data):
    ''' Returns the Manhattan Distance between the centroid
        and the data.
    '''
    distance = sum([abs(x1 - x2) for x1, x2 in zip(centroid, data)])   
    log.debug(f"Manhattan Distance from {centroid} to {data} = {distance}")
    return distance


def calculate_kmedian(dataset):
    ''' Take a list on N dimensions and find the median for each dimension.
    '''
    l = len(dataset)
    temp_sorted_1 = list()
    temp_sorted_2 = list()
    for row in dataset:
        temp_sorted_1.append(row[0])
        temp_sorted_2.append(row[1])
    temp_sorted_1.sort()
    temp_sorted_2.sort()
    # If even, the median is the average of the middle 2 elements
    if l % 2 == 0:
        kmedian1 = (temp_sorted_1[int(l / 2)] + temp_sorted_1[int(l / 2) - 1]) / 2
        kmedian2 = (temp_sorted_2[int(l / 2)] + temp_sorted_2[int(l / 2) - 1]) / 2
    else:
        kmedian1 = temp_sorted_1[int(l / 2) + 1]
        kmedian2 = temp_sorted_2[int(l / 2) + 1]
    log.info(f"K-Median 1: {kmedian1}")
    log.info(f"K-Median 2: {kmedian2}")
    return [kmedian1, kmedian2]


def plot_results(dataset, labels, centroids):
    ''' Plots the dataset and and clusters the points by color.
        It will also plot the calculated k medians as red for visibility.
    '''
    plt.scatter(dataset[:, 0], dataset[:, 1], s=100, c=labels)
    x, y = zip(*centroids)
    plt.scatter(x, y, s=300, c="#FF0000")
    # plt.scatter(centroids[1][0], centroids[1][1], s=30, c=[200])
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Iris Dataset")
    plt.show()


def main():
    args = parse_args()
    log.debug("Start main().")
    np.set_printoptions(precision=10)
    # TODO: use input file
    ## slim it down to 2d,
    X, y = datasets.load_iris(return_X_y=True)
    X = X[:, 0:2]
    log.info(f"Initial Data")
    plt.scatter(X[:, 0], X[:, 1], s=100, c="#000000")
    # TODO: make this input
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Iris Dataset")
    plt.show()

    # log.debug(f"X = {X}")
    # log.debug(f"y={y}")
    ''' Flow:
        initialize_k_centroids()
        assign_kmedian_labels()
        While(sum(k_median_change) > 0.001)
            for each group_by_label
                calculate_new_kmedian_centroid()
                k_median_change[each] = compare_new_and_old_kmedian()
            assign_kmedian_labels()
    '''
    iteration_count = 0
    # Calculate the initial centroids
    centroids = initial_centroids(args.kmedians, X)
    # Make a list of lists, one for each cluster
    groups = [list() for i in range(args.kmedians)]
    # Build a list to track the total movement of the centroids
    centroid_movement = [math.inf for i in range(len(centroids))]
    labels = assign_kmedian_labels(centroids, X)
    plot_results(X, labels, centroids)
    while sum(centroid_movement) > args.epsilon and iteration_count < 100:
        iteration_count += 1
        log.info(f"Iteration Number {iteration_count}")
        # Build each dataset cluster
        for i, label in enumerate(labels):
            log.debug(f"i={i}, label={label}")
            groups[label].append(X[i])
        log.debug(f"Group 1: {groups[0]} \nGroup 2: {groups[1]}")
        # Calculate the centroid for each group
        for i in range(len(centroids)):
            temp_centroid = calculate_kmedian(groups[i])
            # Assign new labels based on the new centroids
            centroid_movement[i] = manhattan_distance(temp_centroid, centroids[i])
            centroids[i] = temp_centroid.copy()
            labels = assign_kmedian_labels(centroids, X)
            temp_centroid.clear()
        # Plot Results for each iteration
        plot_results(X, labels, centroids)
    log.info(f"Completed Calculations: {centroids}")
    plot_results(X, labels, centroids)
    log.debug(f"Compared to Original")
    plot_results(X, y, centroids)
   

if __name__ == "__main__":
    main()