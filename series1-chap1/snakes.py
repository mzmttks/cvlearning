import pickle
import numpy


def isInRange(val, fr, to):
    """Tests if val \in [fr, to] including fr and to"""

    return fr <= val and val <= to


def getNeighbors(pos, data):
    n = []

    for x in range(pos[1]-1, pos[1]+2):
        for y in range(pos[0]-1, pos[0]+2):
            # if x == pos[1] and y == pos[0]:
            #     continue

            if isInRange(x, 0, data.shape[1]-1) and \
               isInRange(y, 0, data.shape[0]-1):
                n.append((y, x))

    for tmpn in n:
        if abs(tmpn[0] - pos[0]) > 1 or abs(tmpn[1] - pos[1]) > 1:
            print "ERROR", n, pos
    return n


def calcEnergy(prevv, curv, nextv, data):

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
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def snakes(data):
    alpha = 1
    beta = 3
    gamma = 2
    # prepare variables
    v = []  # vertex v  (y, x)
    cx = data.shape[1] / 2
    cy = data.shape[0] / 2
    r = min(data.shape[0], data.shape[1]) / 2 - 30

    for theta in range(0, 360, 10):
        rad = theta * numpy.pi / 180
        v.append([int(r * numpy.sin(rad) + cx), int(r * numpy.cos(rad) + cy)])

    N = len(v)  # num of vertice
    dtotal = 0  # total amount of movement
    dbar = 0  # mean inter-vertex distance
    loops = 500  # iteration

    countdown = 10
    prevlenv = N

    # main loop
    pylab.ion()
    for n in range(loops):
        dtotal = 0
        for itmpv, tmpv in enumerate(v):
            energies = []
            neighbors = getNeighbors(tmpv, data)
            for neighbor in neighbors:
                # for rotation
                next = itmpv + 1 if itmpv < len(v) - 1 else 0
                prev = itmpv - 1 if itmpv > 0 else len(v) - 1
                energies.append(calcEnergy(v[prev], neighbor, v[next], data))
                # print prev, itmpv, next, energies[-1]

            # normalize e1, e2
            maxe1 = max(max([e[0] for e in energies]), 1) * 1.0
            maxe2 = max(max([e[1] for e in energies]), 1) * 1.0
            energies = [(e[0] / maxe1, e[1] / maxe2, e[2]) for e in energies]
            energies = [alpha * e[0] + beta * e[1] + gamma * e[2]
                        for e in energies]
            energies = numpy.array(energies)

            # prevent duplicated positions
            for i in range(len(energies)):
                imin = numpy.argmin(energies)
                if not neighbors[imin] in v:
                    nextv = neighbors[imin]
                    break
                else:
                    energies[imin] = numpy.max(energies) + 1

            dtotal += distance(v[itmpv], nextv)
            v[itmpv] = nextv
            # print energies

        # plot
        dmydata = pickle.load(open("simpleimg.pickle"))
        for c in v:
            dmydata[c[1], c[0]] = 200

        # dmydata[v[prev][1],  v[prev][0]] = 100
        # dmydata[v[itmpv][1], v[itmpv][0]] = 100
        # dmydata[v[next][1],  v[next][0]] = 100
        pylab.clf()
        pylab.imshow(dmydata, interpolation="nearest")
        pylab.draw()

        # # remove duplicats
        # newv = [v[0]]
        # for tmpv in v[1:]:
        #     if not tmpv in newv:
        #         newv.append(tmpv)
        # v = newv

        # countdown = countdown -1 if prevlenv == len(v) else 10
        # if countdown == 0:
        #     break

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
