#!/usr/bin/env python
"""
Snakes: A contour detecting algorithm.
       (CVIM Tutorial Series Vol. 1)

(c) T.Mizumoto
"""
import pickle
import itertools
import numpy


def isInRange(val, fr, to):
    """Tests if val \in [fr, to] including fr and to"""

    return fr <= val and val <= to


def getNeighbors(pos, data, excludeSelf=True):
    """Returns 8 neighbors of POS.
    The function ensures that all neighbors are in the shape of the data.
    if excludeSelf is True, the returned neighbors does not include pos itself.
    """

    # get neighbors that
    #   in the shape of the data and
    #   not equals to pos
    neighbors = [n for n
                 in itertools.product(range(pos[0]-1, pos[0] + 2),
                                      range(pos[1]-1, pos[1] + 2))
                 if isInRange(n[1], 0, data.shape[1] - 1) and
                 isInRange(n[0], 0, data.shape[0] - 1) and
                 distance(n, pos) > 0]

    # add pos if excludeSelf is False
    if excludeSelf is False:
        neighbors.append(pos)

    return neighbors


def calcEnergy(prevv, curv, nextv, data):
    """Calculate three energies
       e1: Econt  Tension. Large value is contour is large.
       e2: Ecurv  Curvature. Large value if contour is curved.
       e3: Eimage Edge. Negative large value if the vertex is edge.
       """

    e1 = (nextv[1] - curv[1]) ** 2 + (nextv[0] - curv[0]) ** 2

    tmp1 = nextv[0] - 2 * curv[0] + prevv[0]
    tmp2 = nextv[1] - 2 * curv[1] + prevv[1]
    e2 = tmp1 ** 2 + tmp2 ** 2

    neighbor = [data[t[1], t[0]] for t in getNeighbors(curv, data)]
    imax = max(neighbor)
    imin = min(neighbor)
    e3 = -(data[curv[1], curv[0]] - imin) / max(imax - imin, 1)

    return (e1, e2, e3)


def distance(a, b):
    """Manhattan distance of a and b"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def initVertice(r, offset, theta, margin):
    """
    initializes circular vertice.
    r is a radius, offset is the center,
    theta is a interval in degress,
    and margin is the margin from the edge.
    """
    r -= margin
    radians = [rad * numpy.pi / 180 for rad in range(0, 360, theta)]
    y = [int(r * numpy.sin(rad) + offset[1]) for rad in radians]
    x = [int(r * numpy.cos(rad) + offset[0]) for rad in radians]

    return zip(y, x)


def snakes(data, alpha=1, beta=1, gamma=1, loop=100, isDebug=True):
    # prepare variables
    v = initVertice(min(data.shape[0], data.shape[1])/2,  # radius
                    [data.shape[1]/2, data.shape[0]/2],  # center
                    20,  # theta interval
                    10)  # margin

    dtotal = 0  # total amount of movement
    dbar = 0  # mean inter-vertex distance

    # main loop
    if isDebug:
        pylab.ion()
    for n in range(loop):
        dtotal = 0
        for itmpv, tmpv in enumerate(v):
            # calculate energies for each neighbor
            neighbors = getNeighbors(tmpv, data)
            next = itmpv + 1 if itmpv < len(v) - 1 else 0
            prev = itmpv - 1 if itmpv > 0 else len(v) - 1
            energies = [calcEnergy(v[prev], neighbor, v[next], data)
                        for neighbor in neighbors]

            # normalize e1, e2
            maxe1 = max(max([e[0] for e in energies]), 1) * 1.0
            maxe2 = max(max([e[1] for e in energies]), 1) * 1.0
            energies = [(e[0] / maxe1, e[1] / maxe2, e[2]) for e in energies]

            # calculate energy
            energies = numpy.array([alpha * e[0] + beta * e[1] + gamma * e[2]
                                    for e in energies])

            # Find the next tmpv position,
            # preventing to go to the duplicated positions.
            for i in range(len(energies)):
                imin = numpy.argmin(energies)
                if not neighbors[imin] in v:
                    nextv = neighbors[imin]
                    break
                else:
                    energies[imin] = numpy.max(energies) + 1

            dtotal += distance(v[itmpv], nextv)
            v[itmpv] = nextv

        # plot
        if isDebug:
            dmydata = pickle.load(open("simpleimg.pickle"))
            # for c in v:
            #     dmydata[c[1], c[0]] = 200
            print v
            pylab.clf()
            pylab.plot([tmp[1] for tmp in v], [tmp[0] for tmp in v], "ro-")
            pylab.imshow(dmydata, interpolation="nearest")
            pylab.draw()

    return v

if __name__ == "__main__":
    import pickle
    import pylab

    data = pickle.load(open("simpleimg.pickle"))

    contour = snakes(data)
    for c in contour:
        data[c[1], c[0]] = 2
    pylab.imshow(data, interpolation="nearest")
    pylab.show()
    raw_input()
