''' Calculate K Medians from CSV values

    TODO:
    - Ingest csv files
'''



import math
import random
import numpy as np
from sklearn.datasets import fetch_openml
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
        '-p',
        '--plot',
        required=False,
        type=bool,
        help='Pass any value after the -p option to plot the data.',
        default=False
    )
    argp.add(
        '-s',
        '--dataset',
        required=False,
        type=str,
        help='The name of the dataset to fetch from openml.',
        default='iris'
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
    dimensions = len(dataset[0])
    log.debug(f"Initializing k={k} Centroids from "
              f"dataset size={dataset_size}, dimensions={dimensions}.")
    centroids = list()
    group_size = int(len(dataset)/k + 1)
    log.debug(f"The size of each group is {group_size}")
    shuffled_data = dataset.copy()
    random.shuffle(shuffled_data)
    for i in range(0, dataset_size, group_size):
        log.debug(f"i={i}")
        index = random.randint(i, min(i + group_size, dataset_size - 1)) 
        log.debug(f"Index of centroid: {index}")
        centroids.append(shuffled_data[index])
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
    distance = 0
    for i in range(len(data)):
        distance += abs(data[i] - centroid[i])
    log.debug(f"Manhattan Distance from {centroid} to {data} = {distance}")
    return distance


def calculate_kmedian(dataset):
    ''' Take a list on N dimensions and find the median for each dimension.
    '''
    n = len(dataset[0])
    l = len(dataset)
    temp_sorted = list()
    kmedian = list()
    for i in range(n):
        temp_sorted.append(list())
    for row in dataset:
        for i in range(n):
            temp_sorted[i].append(row[i])
    for i in range(n):
        temp_sorted[i].sort()
    if l % 2 == 0:
        for i in range(n):
            result = (temp_sorted[i][int(l / 2)] + temp_sorted[i][int(l / 2) - 1]) / 2
            kmedian.append(result)
    else:
        for i in range(n):
            result = temp_sorted[i][int(l / 2) + 1]
            kmedian.append(result)
    log.info(f"Calculated k-median = {kmedian}")
    return kmedian


def plot_results(dataset, labels, centroids):
    ''' Plots the dataset and and clusters the points by color.
        It will also plot the calculated k medians as red for visibility.
    '''
    plt.scatter(dataset[:, 0], dataset[:, 1], s=100, c=labels)
    if centroids:
        # zipped = zip(*centroids)
        zipped = np.array(centroids)
        plt.scatter(zipped[:, 0], zipped[:, 1], s=300, c="#FF0000")
        #plt.scatter(centroids[1][0], centroids[1][1], s=30, c=[200])
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Iris Dataset")
    plt.show()


def main():
    args = parse_args()
    log.debug("Start main().")
    np.set_printoptions(precision=10)
    # TODO: use input file
    X, y = fetch_openml(args.dataset, as_frame=False, return_X_y=True)
    name_list = y.copy()
    # random.shuffle()
    log.info(X)
    name_set = list()  # Index mapped to name of classifier
    for a in y:
        if name_set.count(a) == 0:
            name_set.append(a)
    # log.info(f"Name Set: {name_set}")
    for i in range(len(y)):
        y[i] = name_set.index(y[i])
    log.info(y)
    # for name in names:
    # log.info(data)
    # X, y = data['data']
    X = X[:, 0:args.dimensions]
    log.info(f"{X}, {y}")
    # log.info(f"Using {data.details}")    #the {data.Name} dataset, version {data.version}.")
    # X, y = data.load_digits(return_X_y=True)
    # X = X[:, 0:args.dimensions]
    log.info(f"Initial Data")
    # If plotting, then show the initial data set
    if args.plot:
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
    if args.plot:
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
        if args.plot:
            plot_results(X, labels, centroids)
    log.info(f"Completed Calculations: {centroids}")
    if args.plot: plot_results(X, labels, centroids)
    log.info(f"Compared to Original")
    if args.plot: plot_results(X, y, list())
    correct_label = 0
    # for i, x in enumerate(X):

    
    for a, b in zip(y, labels):
        # log.info(f"{a}, {b}")
        if a == b:
            correct_label += 1
    log.info(f"Percent Correct: {100*correct_label/len(y)}")
   

if __name__ == "__main__":
    main()