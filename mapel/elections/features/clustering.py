
import numpy as np


def clustering_v1(experiment, num_clusters=20):
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    import scipy.spatial.distance as ssd

    # skip the paths

    SKIP = ['UNID', 'ANID', 'STID', 'ANUN', 'STUN', 'STAN',
            'Mallows',
            'Urn',
            'Identity', 'Uniformity', 'Antagonism', 'Stratification',
            ]

    new_names = []
    for i, a in enumerate(list(experiment.distances)):
        if not any(tmp in a for tmp in SKIP):
            new_names.append(a)
    print(len(new_names))

    distMatrix = np.zeros([len(new_names), len(new_names)])
    for i, a in enumerate(new_names):
        for j, b in enumerate(new_names):
            if a != b:
                distMatrix[i][j] = experiment.distances[a][b]

    # Zd = linkage(ssd.squareform(distMatrix), method="complete")
    # cld = fcluster(Zd, 500, criterion='distance').reshape(len(new_names), 1)

    Zd = linkage(ssd.squareform(distMatrix), method="complete")
    cld = fcluster(Zd, 12, criterion='maxclust').reshape(len(new_names), 1)

    clusters = {}
    for i, name in enumerate(new_names):
        clusters[name] = cld[i][0]
    for name in experiment.coordinates:
        if name not in clusters:
            clusters[name] = 0
    return {'value': clusters}


def clustering_kmeans(experiment, num_clusters=20):
    from sklearn.cluster import KMeans

    points = list(experiment.coordinates.values())

    kmeans = KMeans(n_clusters=num_clusters)

    kmeans.fit(points)

    y_km = kmeans.fit_predict(points)

    # plt.scatter(points[y_km == 0, 0], points[y_km == 0, 1], s=100, c='red')
    # plt.scatter(points[y_km == 1, 0], points[y_km == 1, 1], s=100, c='black')
    # plt.scatter(points[y_km == 2, 0], points[y_km == 2, 1], s=100, c='blue')
    # plt.scatter(points[y_km == 3, 0], points[y_km == 3, 1], s=100, c='cyan')

    # all_distances = []
    # for a,b in combinations(experiment.distances, 2):
    #     all_distances.append([a, b, experiment.distances[a][b]])
    # all_distances.sort(key=lambda x: x[2])
    #
    # clusters = {a: None for a in experiment.distances}
    # num_clusters = 0
    # for a,b,dist in all_distances:
    #     if clusters[a] is None and clusters[b] is None:
    #         clusters[a] = num_clusters
    #         clusters[b] = num_clusters
    #         num_clusters += 1
    #     elif clusters[a] is None and clusters[b] is not None:
    #         clusters[a] = clusters[b]
    #     elif clusters[a] is not None and clusters[b] is None:
    #         clusters[b] = clusters[a]
    clusters = {}
    for i, name in enumerate(experiment.coordinates):
        clusters[name] = y_km[i]
    return {'value': clusters}


def id_vs_un(experiment, election_ids, feature_params=None):
    results = {}
    experiment.distances['ID']['ID'] = 0
    experiment.distances['UN']['UN'] = 0
    for election_id in election_ids:
        distance_to_id = experiment.distances['ID'][election_id]
        distance_to_un = experiment.distances['UN'][election_id]
        if distance_to_id > distance_to_un:
            results[election_id] = 1
        elif distance_to_id < distance_to_un:
            results[election_id] = 0
        else:
            results[election_id] = 0.5
    return results


def an_vs_st(experiment, election_ids, feature_params=None):
    results = {}
    experiment.distances['AN']['AN'] = 0
    experiment.distances['ST']['ST'] = 0
    for election_id in election_ids:
        distance_to_an = experiment.distances['AN'][election_id]
        distance_to_st = experiment.distances['ST'][election_id]
        if distance_to_an > distance_to_st:
            results[election_id] = 1
        elif distance_to_an < distance_to_st:
            results[election_id] = 0
        else:
            results[election_id] = 0.5
    return results
