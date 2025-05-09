import numpy as np

"""
Compute social discomfort generated 
when having the robot at (x,y) position
By Kirby: https://www.ri.cmu.edu/pub_files/2010/5/rk_thesis.pdf

"""
def kirby(xp,yp,thetap,x,y):
    phih = 2  # variable
    phir = phih * (1 / 2)  # variable
    phis = phih * (2 / 3)  # variable
    theta = thetap
    
    xd = x - xp
    yd = y - yp
    alpha = np.arctan2(yd, xd) - theta + np.pi / 2
    
    # Normalizing alpha
    if alpha >= np.pi:
        alpha -= 2 * np.pi
    elif alpha < -np.pi:
        alpha += 2 * np.pi
    
    # Selecting phi 
    phi = phir if alpha <= 0 else phih
    
    a = ((np.cos(theta)**2) / (2 * phi**2)) + ((np.sin(theta)**2) / (2 * phis**2))
    b = (np.sin(2 * theta) / (4 * phi**2)) - (np.sin(2 * theta) / (4 * phis**2))
    c = ((np.sin(theta)**2) / (2 * phi**2)) + ((np.cos(theta)**2) / (2 * phis**2))
    
    unDegree = np.exp(-(a * xd**2 + 2 * b * xd * yd + c * yd**2))
    
    # When phih = 2, 0.2 is approximately 3.6 distance from the front.
    if unDegree <= 0.2:
        unDegree = 0
    
    return unDegree

"""
Discretize a continuous observation and convert to a single index. (fixed_target)
"""
def obs_to_index(obs, bins, low, high):
    ratios = (obs - low) / (high - low)
    ratios = np.clip(ratios, 0, 0.999)  # avoid going out of bounds
    discrete_obs = (ratios * bins).astype(int)
    return np.ravel_multi_index(discrete_obs, bins)

"""
Discretize a continuous observation and convert to a single index. (any_target)
"""
def discretize_obs(obs, agent_edges, target_edges, agent_bins, target_bins):    

    # Discretizing
    agent_disc = [np.digitize(obs['agent'][i], agent_edges[i]) for i in range(len(agent_edges))]
    target_disc = [np.digitize(obs['target'][i], target_edges[i]) for i in range(len(target_edges))]
    
    # Convert multidimensional indices to flat index (1-table)
    full_bins = np.concatenate((agent_bins, target_bins))
    indices = agent_disc + target_disc
    flat_index = np.ravel_multi_index(indices, full_bins)
    return flat_index

"""
Create bins for discretization. (any_target)
"""
def create_bins(low, high, bins):
    return [np.linspace(l, h, b + 1)[1:-1] for l, h, b in zip(low, high, bins)]