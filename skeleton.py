import mapel
from math import sqrt


def clear():
    for i in range(5):
        print()

def line():
    print("--------------------------------------------------------")



def euclid_diagonal( experiment ):
    coordinates = experiment.coordinates
    P1 = "ID"
    P2 = "UN"
    x1 = coordinates[P1][0]
    y1 = coordinates[P1][1]
    x2 = coordinates[P2][0]
    y2 = coordinates[P2][1]
    D = sqrt( (x2-x1)**2 + (y2-y1)**2 )
    return D


def poswise_diagonal( experiment ):
    distances = experiment.distances
    P1 = "ID"
    P2 = "UN"
    return distances[P1][P2]


def fix_distances( dist, names ):
    for P1 in names:
        for P2 in names:
            if P1 == P2:
                dist[P1][P2] = 0
                dist[P2][P1] = 0
            elif P2 not in dist[P1]:
                dist[P1][P2] = dist[P2][P1]


def print_coordinates( experiment, names ):
    coordinates = experiment.coordinates
    for P in names:
        print(P, "-->", coordinates[P][0], coordinates[P][1] )



def normalize_coordinates( experiment ):
    pass



def print_table( T, names, field_len ):

    # column names
    print("".ljust( field_len ),end="")
    for P1 in names:
        print(P1.ljust( field_len ),end="")
    print()

    # rows
    for P1 in names:
        print(P1.ljust( field_len ),end="")
        for P2 in names:
            if type(T[P1][P2]) == str:
                print( T[P1][P2].ljust( field_len ), end="" )
            elif type(T[P1][P2]) in [int,float]:
                print( ("%.2f" % T[P1][P2]).ljust( field_len ), end="")
        print()
    

    


def print_tables( experiment ):
    clear()
    namez = ["UN","ID","AN","ST","CON","WAL", "CAT"]  # MAL
    namez += ["MAL0%d" % i for i in range(1,10) ]
    names = []
    for n in namez:
        names.append( n ) # I+"_0")
    fix_distances( experiment.distances, names )

    line()
    print_coordinates( experiment, names )
    line()


    euclid_D  = euclid_diagonal ( experiment )
    poswise_D = poswise_diagonal( experiment )
    dist = { n : dict() for n in names }
    diff = { n : dict() for n in names }
    posw = { n : dict() for n in names }
    eucd = { n : dict() for n in names }
    for P1 in names:
        for P2 in names:
            dist[P1][P2]="--"
            diff[P1][P2]="--"
            posw[P1][P2]="--"
            eucd[P1][P2]="--"

    distances   = experiment.distances
    coordinates = experiment.coordinates
    diff_min = dist_min = 100
    diff_max = dist_max = -100
    for P1 in names:
        for P2 in names:
            if P1 == P2: continue
            x1 = coordinates[P1][0]
            y1 = coordinates[P1][1]
            x2 = coordinates[P2][0]
            y2 = coordinates[P2][1]
            euclidean = float( sqrt( (x2-x1)**2 + (y2-y1)**2 ) / euclid_D )
            poswise   = float( distances[P1][P2] / poswise_D )
            posw[P1][P2] = poswise
            eucd[P1][P2] = euclidean
            print(P1,P2, poswise)
            dist[P1][P2] = euclidean / poswise
            diff[P1][P2] = euclidean - poswise
            diff_min = min( diff_min, diff[P1][P2] )
            diff_max = max( diff_max, diff[P1][P2] )
            dist_min = min( dist_min, dist[P1][P2] )
            dist_max = max( dist_max, dist[P1][P2] )
#            dist[P1][P2] = "x"

    clear()
    line()
    print("Relative differences")
    print("min =", dist_min, "   max =", dist_max )
    line()
    print_table( dist, names, 8 )
    line()        


    clear()
    line()
    print("Absolute differences")
    print("min =", diff_min, "   max =", diff_max )
    line()
    print_table( diff, names, 8 )
    line()        


    clear()
    line()
    print("Positionwise Distances")
    line()
    print_table( posw, names, 8 )
    line()        

    clear()
    line()
    print("Euclidean Distances")
    line()
    print_table( eucd, names, 8 )
    line()        





if __name__ == "__main__":
    m = 16
    n = 100

    experiment = mapel.prepare_experiment()

    experiment.set_default_num_candidates( m )
    experiment.set_default_num_voters( n )

    # Compass Matrices
    experiment.add_election(election_model="uniformity", election_id="UN", color = (1,0.5,0.5), marker="X")
    experiment.add_election(election_model="identity", election_id="ID", color="red", marker="X")
    experiment.add_election(election_model="antagonism", election_id="AN", color="black", marker = "o")
    experiment.add_election(election_model="stratification", election_id="ST", color="black")

    # Paths
    base = 30
    unid = base-2
    anid = int(0.75*base)-2
    stun = int(0.75*base)-2
    anun = int(0.5*base)-2
    stid = int(0.5*base)-2
    stan = int(13.0/16.0*base)-2
    experiment.add_family(election_model='anid', size=anid, color='gray', marker=".")
    experiment.add_family(election_model='stid', size=stid, color='gray', marker=".")
    experiment.add_family(election_model='anun', size=anun, color='gray', marker=".")
    experiment.add_family(election_model='stun', size=stun, color='gray', marker=".")
#    experiment.add_family(election_model='unid', size=unid, color='gray', param_1=4)
#    experiment.add_family(election_model='stan', size=stan, color='gray', param_1=4)


    # Single-Peaked
    experiment.add_election(election_model="conitzer_matrix", election_id="CON", color="blue")
    experiment.add_election(election_model="walsh_matrix", election_id="WAL", color="cyan")
    
    # Caterpillar
    experiment.add_election(election_model="gs_caterpillar_matrix", election_id="CAT", color="pink")

    experiment.add_election(election_model="sushi_matrix", election_id="SHI", color="green")
    experiment.add_election(election_model="2d_grid", election_id="SHI", color="magenta")
#    experiment.add_election(election_model="single-crossing_matrix", election_id="SCR", color="gray")


    # Mallows
    MAL_COUNT = 10
    for i in range(1,MAL_COUNT):
        normphi = 1.0/MAL_COUNT*i
        experiment.add_election(election_model="norm-mallows_matrix", election_id=("MAL0%d"% i), color=(normphi,normphi,1), params = {"norm-phi": normphi})


    experiment.compute_distances()
    for n in [30,60,90]:
        experiment.embed( algorithm="lle", num_neighbors = n)
    # normalize_coordinates( experiment )

    # print_tables( experiment )
    
        experiment.print_map(title='Skeleton Map', saveas='skeleton', ms=30, legend=True,
                             mixed=True)




