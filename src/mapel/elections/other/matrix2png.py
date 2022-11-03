from PIL import Image, ImageDraw

import mapel


def getrgb(value, MAX):
    x = int(255 * value / MAX)
    return (x, x, x)


def getrgb_uniform(value, MAX):
    x = int(255 * value)
    return (x, x, x)


def getsqrtrgb(value, MAX):
    x = int(255 * (value ** 0.33) / (MAX ** 0.33))
    return (x, x, x)


def getsqrtrgb_uniform(value, MAX):
    x = int(255 * (value ** 0.25))
    return (x, x, x)


def matrix2png(argv):
    # introduce yourself
    if len(argv) < 4:
        print("Invocation:")
        print("  python3 matrix2png num_candidates culture_id reorder [param1]")
        print(
            "  reorder -- election_id of the culture_id to try to resemble (e.g., ID, or AN); use org to use original order")
        print("")
        exit()

    # gather arguments
    m = int(argv[1])
    n = m * m
    model = argv[2]
    tgt = argv[3]
    print("TGT:", tgt)
    if len(argv) >= 5:
        param = float(argv[4])
    else:
        param = None

    if model != "mallows":
        name = "%s_%d_%s.png" % (model, m, tgt)
    else:
        name = "%s_phi%d_%d_%s.png" % (model, param * 100, m, tgt)

    # prepare the experiment/matrix
    experiment = mapel.prepare_experiment()
    experiment.set_default_num_candidates(m)
    experiment.set_default_num_voters(n)

    # Compass Matrices
    experiment.add_election(election_model="uniformity", election_id="UN", color=(1, 0.5, 0.5),
                            marker="X")
    experiment.add_election(election_model="identity", election_id="ID", color="red", marker="X")
    experiment.add_election(election_model="antagonism", election_id="AN", color="black",
                            marker="o")
    experiment.add_election(election_model="stratification", election_id="ST", color="black")

    # form the matrix
    if model != "mallows":
        experiment.add_election(election_model=model, election_id="M")
    else:
        experiment.add_election(election_model="norm-mallows_matrix", params={"norm-phi": param},
                                election_id="M")
    M = experiment.elections["M"].matrix

    # get the mapping to a given election
    experiment.compute_distances()
    if tgt == "org":
        match = list(range(m))
    else:
        match = experiment.matchings[tgt]["M"]
    print(match)
    # get reversed matching
    rev_match = [0] * m
    for i in range(m):
        rev_match[match[i]] = i
    print(rev_match)

    # create the image
    img = Image.new("RGB", (m, m), color="black")
    draw = ImageDraw.Draw(img)

    MAX = 0  # highest value in the matrix
    for y in range(m):
        for x in range(m):
            MAX = max(MAX, M[y][x])

    color = lambda v: getsqrtrgb_uniform(v, MAX)

    ### print columns
    print("----")
    for x in range(m):
        print("%.2f" % x, end=" ")
    print()
    print("----")

    ### draw the matrix
    for y in range(m):
        for x in range(m):
            draw.point((x, y), fill=color(M[y][rev_match[x]]))
            print("%.2f" % M[y][rev_match[x]], end=" ")
        print()

    # save the image
    img.save(name)

    print("MAX value:", MAX)
