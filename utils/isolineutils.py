import numpy as np
import sys
sys.path.insert(0, '/home/dpaiton/DeepSparseCoding/')
import utils.data_processing as dp

def get_isoline_areas(trials, num_points=1000, threshold=np.sqrt(-2*np.log(1-0.68)),
    ellipse_tolerance=0.01):
    """
    Wrapper function for getting isoline areas from probability maps.
    subj_maps is the output of get_density_maps
    gauss_mean and gauss_cov are returned from the get_gauss_fit() function in
        utils/data_processing.py, which is from the github.com/dpaiton/DeepSparseCoding repsitory
    """
    ms_indices = [idx for idx, trial in enumerate(trials) if trial.sub_ms=="1"]
    control_indices = [idx for idx, trial in enumerate(trials) if trial.sub_ms=="0"]
    subj_maps, delta_position = get_density_maps(trials, num_points)
    patient_areas = []
    control_areas = []
    for trace_idx, prob_map in enumerate(subj_maps):
        gauss_fit, grid, gauss_mean, gauss_cov = dp.get_gauss_fit(prob_map, num_attempts=1)
        keep_points, throw_points = find_thresholded_points(prob_map, gauss_mean, gauss_cov,
            threshold)
        isoline_points = np.zeros_like(prob_map)
        isoline_points[keep_points] = 1
        ellipse_center, ellipse_radii, ellipse_rotation = get_min_area_ellipse(isoline_points,
            ellipse_tolerance)
        ellipse_area = np.pi * np.prod(ellipse_radii*delta_position)
        if trace_idx in ms_indices:
            patient_areas.append(ellipse_area)
        elif trace_idx in control_indices:
            control_areas.append(ellipse_area)
        else:
            assert False
    return (patient_areas, control_areas)

def get_density_maps(trials, num_points=1000):
    """
    Estimate probability density function using histogram analysis
    """
    trial_xs = [trial.x for trial in trials]
    trial_ys = [trial.y for trial in trials]
    abs_min = np.amin([np.amin([np.amin(trial_x) for trial_x in trial_xs]),
        np.min([np.amin(trial_y) for trial_y in trial_ys])])
    abs_max = np.amax([np.max([np.amax(trial_x) for trial_x in trial_xs]),
        np.max([np.amax(trial_y) for trial_y in trial_ys])])
    vector_points = np.linspace(abs_min, abs_max, num_points+1)
    delta_position = (abs_max-abs_min)/(num_points+1) # resolution
    subj_maps = []
    for trace_idx in range(len(trials)):
        # do log?
        hist, x_edges, y_edges = np.histogram2d(trial_xs[trace_idx], trial_ys[trace_idx],
            bins=vector_points)
        hist /= np.max(hist)
        subj_maps.append(hist)
    return (subj_maps, delta_position)

def find_thresholded_points(prob_map, gauss_mean, gauss_cov, threshold=np.sqrt(-2*np.log(1-0.68))):
    """
    Following:
        https://en.wikipedia.org/wiki/Mahalanobis_distance
    default threshold should be a level of density that corresponds to ~68% of the data points
    gauss_mean and gauss_cov are returned from the get_gauss_fit() function in
        utils/data_processing.py, which is from the github.com/dpaiton/DeepSparseCoding repsitory

    TODO: implement pseudo-code from here: https://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440

    Input: A 2x10 matrix P storing 10 2D points and tolerance = tolerance for error.
    Output: The equation of the ellipse in the matrix form, i.e. a 2x2 matrix A and a 2x1 vector C representing the center of the ellipse.

       // Dimension of the points d = 2;
       // Number of points N = 10;

       // Add a row of 1s to the 2xN matrix P - so Q is 3xN now. Q = [P;ones(1,N)]

       // Initialize count = 1; err = 1; //u is an Nx1 vector where each element is 1/N u = (1/N) * ones(N,1)

       // Khachiyan Algorithm while err > tolerance { // Matrix multiplication: // diag(u) : if u is a vector, places the elements of u // in the diagonal of an NxN matrix of zeros X = Qdiag(u)Q'; // Q' - transpose of Q

       // inv(X) returns the matrix inverse of X
       // diag(M) when M is a matrix returns the diagonal vector of M
       M = diag(Q' * inv(X) * Q); // Q' - transpose of Q  

       // Find the value and location of the maximum element in the vector M
       maximum = max(M);
       j = find_maximum_value_location(M);

       // Calculate the step size for the ascent
       step_size = (maximum - d -1)/((d+1)*(maximum-1));

       // Calculate the new_u:
       // Take the vector u, and multiply all the elements in it by (1-step_size)
       new_u = (1 - step_size)*u ;

       // Increment the jth element of new_u by step_size
       new_u(j) = new_u(j) + step_size;

       // Store the error by taking finding the square root of the SSD 
       // between new_u and u
       // The SSD or sum-of-square-differences, takes two vectors 
       // of the same size, creates a new vector by finding the 
       // difference between corresponding elements, squaring 
       // each difference and adding them all together. 

       // So if the vectors were: a = [1 2 3] and b = [5 4 6], then:
       // SSD = (1-5)^2 + (2-4)^2 + (3-6)^2;
       // And the norm(a-b) = sqrt(SSD);
       err = norm(new_u - u);

       // Increment count and replace u
       count = count + 1;
       u = new_u;
       }

       // Put the elements of the vector u into the diagonal of a matrix // U with the rest of the elements as 0 U = diag(u);

       // Compute the A-matrix A = (1/d) inv(P U P' - (P u)(Pu)' );

       // And the center, c = P * u;
    """
    # index of each point with >0 probability
    prob_points = np.stack(np.where(prob_map>0), axis=0) # shape = [2, num_points]
    # go through each observation and determine distance from the mean, scaled by the variance
    throw_points = []
    keep_points = []
    for obs_idx in range(prob_points.shape[1]):
        observation = prob_points[:,obs_idx][:,None]
        mdist = np.squeeze(np.sqrt(np.dot(np.dot((observation-gauss_mean[:,None]).T, np.linalg.inv(gauss_cov)),
            (observation-gauss_mean[:,None]))))
        if mdist < threshold:
            throw_points.append(np.squeeze(observation))
        else:
            keep_points.append(np.squeeze(observation))
    # Need to reorder lists for numpy indexing:
    #     http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing
    throw_points = tuple(zip(*throw_points))
    keep_points = tuple(zip(*keep_points))
    return (keep_points, throw_points)

def get_min_area_ellipse(point_map, tolerance=0.01):
    """
    Find the minimum volume ellipse which holds all the points
    Adapted from https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.116.7691
    P is a numpy array of shape [num_points, 2]
    Returns:
    (center, radii, rotation)
    """
    P = np.stack(np.where(point_map>0), axis=1) # shape = [num_points, 2]
    (N, d) = np.shape(P)
    d = float(d) # dimensionality of points
    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)])  # 3xN
    QT = Q.T
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)
    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(np.linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
    # center of the ellipse
    center = np.dot(P.T, u)
    # the A matrix for the ellipse
    A = np.linalg.inv(
        np.dot(P.T, np.dot(np.diag(u), P)) -
        np.array([[a * b for b in center] for a in center])) / d
    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0/np.sqrt(s)
    return (center, radii, rotation)
