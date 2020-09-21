from matplotlib import pyplot as plt


def vis(X, name):
    plt.plot(X.T)
    plt.title(f"{X.shape[0]} stocks from {name}")
    plt.show()


def plot_multiplier(algs):
    # algs- list where algs[i] == [data(np array), name(string)]
    T = len(algs[0][0])  # steps

    for alg in algs:
        plt.plot(alg[0], label=alg[1])

    plt.legend()
    plt.xlabel("time")
    plt.ylabel("log of stock multiplier")
    plt.show()


def plot_regret(hind, algs):
    # hind==hindsight- np array
    # algs as in plot_multiplier
    T = len(hind)
    for alg in algs:
        plt.plot([(h - a) for h, a in zip(hind, alg[0])], label=alg[1])

    plt.legend()
    plt.xlabel("time")
    plt.ylabel("regret")

    plt.show()
