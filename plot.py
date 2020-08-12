from matplotlib import pyplot as plt

def plot_two_plots(hind,oga,ons):
    T = len(hind)

    # stock multiplier
    plt.subplot(121)
    plt.plot(hind, color = 'b', label="best in hindsight")
    plt.plot(oga, color = 'r', label="oga")
    plt.plot(ons, color = 'c', label="ons")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("log of stock multiplier")

    # regret
    plt.subplot(122)
    plt.plot([(h - o) for h,o in zip(hind,oga)], color = 'r', label="oga")
    plt.plot([(h - o) for h,o in zip(hind,ons)], color = 'c', label="ons")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("regret")

    plt.show()

def vis(X):
    plt.plot(X.T)
    plt.title(f"{X.shape[0]} stocks")
    plt.show()
