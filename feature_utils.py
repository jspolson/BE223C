import math
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import itertools

def bbox2_ND(img):
    '''
    Generates a three-dimensional bounding box given a 3D numpy array.
    '''
    N = img.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    b = tuple(out)
    return b

def generate_features_noncluster(patients, raw_data, tmax_seg = np.arange(5,50,5), tmax_mod = 'TMAXs', proj_mod = ['preflair', 'dwi']):
    '''
    Generates features based on the noncluster method.
    Inputs:
    - patients: list of patient IDs
    - raw_data: raw imaging data
    - tmax_seg: the boundaries from which to generate segments from the tmax (or other specified) modality. Default: np.arange(5,50,5)
    - tmax_mod: the imaging modality to segment. Default: 'TMAXs'
    - proj_mod: the modalities to which the function applies the segmentation masks. Default: ['preflair', 'dwi']
    
    Outputs:
    - featArr: a feature array for each patient. For each modality, and each segmentation, produces the sum (volume), mean, variance, minimum, maximum, centroid coordinates, and distance from total centroid.  
    '''
    features = []

    for pt in patients:

        non_neg = np.where(raw_data[pt][tmax_mod]<0, 0, raw_data[pt][tmax_mod])
        b = bbox2_ND(non_neg)
        non_neg = non_neg[b[4]:b[5],b[2]:b[3],b[0]:b[1]]

        masks = {}
        vol_feat = []
        mean_feat = []
        var_feat = []
        max_feat = []
        min_feat = []
        centroid_feat = []
        dist_feat = []
        for mod in proj_mod:
            proj_im = raw_data[pt][mod][b[4]:b[5],b[2]:b[3],b[0]:b[1]]

            for i in np.arange(0,len(tmax_seg), 1):
                if i == 0:
                    cat = ("Tmax under " + str(tmax_seg[i]))
                    g1 = np.where(non_neg>=tmax_seg[i], 0, non_neg)

                    masks[cat] = g1 > 0
                else:
                    t1 = tmax_seg[i-1]
                    t2 = tmax_seg[i]
                    cat = ("Tmax between " + str(t1) + " and "+ str(t2))
                    g1 = np.where(non_neg<t1, 0, non_neg)
                    g2 = np.where(g1>=t2, 0, g1)
                    masks[cat] = g2 > 0
                proj = masks[cat] * proj_im
                # calculate the centroid of the projection
                x, y, z = np.nonzero(proj_im)
                cx = np.mean(x)
                cy = np.mean(y)
                cz = np.mean(z)

                #print(proj[proj.nonzero()].shape)

                ##### create the features
                if proj[proj.nonzero()].shape[0] == 0:
                    # volume
                    vol_feat.append(0)
                    # mean
                    mean_feat.append(0)
                    # variance
                    var_feat.append(0)
                    # maximum
                    max_feat.append(0)
                    # minimum
                    min_feat.append(0)
                    #multi_slice_viewer(proj, aspect = 1)

                else: 
                # volume
                    vol_feat.append(np.sum(masks[cat]*1))
                    # mean
                    mean_feat.append(proj[proj.nonzero()].mean())
                    # variance
                    var_feat.append(proj[proj.nonzero()].var())
                    # maximum
                    max_feat.append(proj[proj.nonzero()].max())
                    # minimum
                    min_feat.append(proj[proj.nonzero()].min())
                    #multi_slice_viewer(proj, aspect = 1)

                # centroid
                x, y, z = np.nonzero(proj)
                c1 = np.mean(x)
                c2 = np.mean(y)
                c3 = np.mean(z)
                centroid_feat.append(c1)
                centroid_feat.append(c2)
                centroid_feat.append(c3)

                #distance
                dist = math.sqrt((cx - c1)**2 + (cy - c2)**2 + (cz - c3)**2)
                dist_feat.append(dist)

        feat_list = vol_feat + mean_feat + var_feat + max_feat + min_feat + centroid_feat + dist_feat
        features.append(feat_list)
    featArr = np.stack(tuple(features), axis=0)
    
    return featArr

def generate_features_cluster(patients, raw_data, tmax_seg = [4,8,10,50], tmax_mod = 'TMAXs', proj_mod = ['preflair', 'dwi'], n_clusters =100):
    '''
    Generates features based on the noncluster method.
    Inputs:
    - patients: list of patient IDs
    - raw_data: raw imaging data
    - tmax_seg: the boundaries from which to generate segments from the tmax (or other specified) modality. Default: [4,8,10,50]
    - tmax_mod: the imaging modality to segment. Default: 'TMAXs'
    - proj_mod: the modalities to which the function applies the segmentation masks. Default: ['preflair', 'dwi']
    
    Outputs:
    - featArr: a feature array for each patient. For each modality, and each segmentation, produces the mean, variance, minimum, and maximum.  
    '''
    features = []

    for pt in patients:
        feat_list = []

        non_neg = np.where(raw_data[pt][tmax_mod]<0, 0, raw_data[pt][tmax_mod])
        non_neg = np.where(non_neg > tmax_seg[-1], 0 , non_neg)
        b = bbox2_ND(non_neg)
        non_neg = non_neg[b[4]:b[5],b[2]:b[3],b[0]:b[1]]

        masks = {}

        total_vol = np.sum(non_neg > 0 *1)
        clust_vol = round(total_vol/n_clusters,0)

        for i in np.arange(0,len(tmax_seg), 1):
            if i == 0:
                cat = ("Tmax under " + str(tmax_seg[i]))
                g1 = np.where(non_neg>tmax_seg[i], 0, non_neg)

                masks[cat] = g1 > 0
            else:
                t1 = tmax_seg[i-1]
                t2 = tmax_seg[i]
                cat = ("Tmax between " + str(t1) + " and "+ str(t2))
                g1 = np.where(non_neg<t1, 0, non_neg)
                g2 = np.where(g1>t2, 0, g1)
                masks[cat] = g2 > 0


        for mod in proj_mod:
            cluster_numbers = []
            coordinates = []
            proj_im = raw_data[pt][mod][b[4]:b[5],b[2]:b[3],b[0]:b[1]]
            for m in masks:
                proj = masks[m] * proj_im

                # calculate the centroid of the projection
                coords = np.asarray(np.nonzero(proj)).T
                n_clust = int(round(coords.shape[0]/clust_vol, 0))

                cluster_numbers.append(n_clust)
                coordinates.append(coords)

            cluster_numbers = [x + (n_clusters -sum(cluster_numbers)) if x==max(cluster_numbers) else x for x in cluster_numbers]

            for n, c in zip(cluster_numbers, coordinates):
                if n == 0:
                    pass
                else:
                    clusters = MiniBatchKMeans(int(n)).fit_predict(c.astype(int))

                    clust_ind = [np.where(clusters == i)[0] for i in np.unique(clusters)]

                    clust_coords = [c[i] for i in clust_ind]

                    for clust in clust_coords:
                        signals = [proj_im[p[0],p[1],p[2]] for p in clust]
                        feat_list.append(np.asarray(signals).mean())
                        feat_list.append(np.asarray(signals).var())
                        feat_list.append(np.asarray(signals).max())
                        feat_list.append(np.asarray(signals).min())
        features.append(feat_list)
    featArr = np.stack(tuple(features), axis=0)
    
    return featArr
