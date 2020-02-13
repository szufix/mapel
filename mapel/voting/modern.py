import matplotlib.pyplot as plt
import numpy as np
import voting.objects as obj


def print_2d(name, num_winners=0, shades=False):
    model = obj.Model_2d(name, num_winners=num_winners, shades=shades)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if shades:
        file_name = "results/times/" + str(name) + ".txt"
        file_ = open(file_name, 'r')
        ctr = 0
        for k in range(model.num_families):
            for _ in range(model.family_size[k]):
                if ctr < 200:
                    shade = float(file_.readline())
                else:
                    shade = 1.
                ax.scatter(model.points[ctr][0], model.points[ctr][1],
                           color=model.colors[k], alpha=model.alphas[k]*shade, s=9)
                ctr += 1

    else:
        #for w in model.winners:
        #for w in [513, 527]:
        #    ax.scatter(model.points[w][0], model.points[w][1], color="black", s=30)
        for k in range(model.num_families):
            ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                       color=model.colors[k], label=model.labels[k],
                       alpha=model.alphas[k], s=9)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    name = "images/" + str(name) + ".png"
    plt.savefig(name)
    plt.show()


