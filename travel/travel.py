import os
import argparse
import sys

import numpy as np
from pandas import DataFrame, Series


prices_file = os.path.join(os.path.dirname(__file__), 'prices.csv')
PRICES = DataFrame.from_csv(prices_file).T.astype(float)


def memoize(f):
    memory = {}

    def mem(*args):
        try:
            res = memory[args]
        except KeyError:
            res = f(*args)
            memory[args] = res
        return res
    return mem


def levenshtein(a, b):
    @memoize
    def lev(i, j):
        if min(i, j) == 0:
            return max(i, j)
        else:
            return min(lev(i-1, j) + 1,
                       lev(i, j-1) + 1,
                       lev(i-1, j-1) + (0 if a[i-1] == b[j-1] else 1))

    return lev(len(a), len(b))


def get_correct_city(bad_name):
    return (Series(PRICES.index.map(lambda s: levenshtein(bad_name, s)),
                   index=PRICES.index)
            .idxmin())


def rebuild_path(prev, from_, to):
    curr = to
    path = []
    while curr != from_:
        path.append(curr)
        curr = prev[curr]
    path.append(from_)
    return path[::-1]


def get_path(from_, to):
    # dijkstra!
    cities = PRICES.index
    costs = Series(index=cities).fillna(np.inf)
    costs[from_] = 0
    curr = from_
    unvisited = cities.copy()
    prev = {}
    while to in unvisited:
        unvisited = unvisited.drop(curr)
        unvisited_neighbors = PRICES[curr][unvisited].dropna()
        tentative_distances = unvisited_neighbors + costs[curr]
        if tentative_distances.min() == np.inf:
            break  # unreachable
        else:
            for node, tentative in tentative_distances.items():
                if tentative < costs[node]:
                    costs[node] = tentative
                    prev[node] = curr
        curr = costs[unvisited].idxmin()
        if curr == to:
            break

    return rebuild_path(prev, from_, to), costs[to]


def main(args):
    from_ = get_correct_city(args.start)
    to = get_correct_city(args.finish)

    path, cost = get_path(from_, to)
    print('Path from {} to {} ({} gold):'.format(from_, to, cost))
    print(' -> '.join(path))


def parse_args():
    parser = argparse.ArgumentParser('Find the cheapest path between two towns')
    parser.add_argument('start')
    parser.add_argument('finish')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

