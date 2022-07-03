import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import animation
import statsmodels.formula.api as smf
import warnings
warnings.simplefilter("ignore")

##############################################################
class Particle:
    """ 
    This class is used to initialize the particles. 
    It holds the position (translational and rotational) and connections and the respective rest lengths
    It can also used to copy a particle class by providing another particle, as copying a class without codependencies is difficult.
    """
    def __init__(self, position = None, rotations = np.zeros(1), connected_to = [], force = np.zeros((2,1)).astype(float), torque = 0, other = None):
        if other:
            self.position = other.position.copy()
            self.rotations = other.rotations.copy()
            self.connected_to = other.connected_to.copy()
            self.force = other.force.copy()
            self.torque = other.torque
            self.n_spins = other.n_spins.copy()
            self.rest_lengths = other.rest_lengths.copy()
        else:
            self.position = position # the position of the particle
            self.rotations = rotations # the rotations of the particle to their respective spring connection
            self.connected_to = connected_to # the list with particles that are connected through a spring
            self.force = force # the current force acting on the particle
            self.torque = torque # the current torque acting on the particle
            self.n_spins = np.zeros(len(connected_to)).astype(float) # to keep count of the rotations the particle, relative with other particles makes, as radians does not hold count of rotations
            self.rest_lengths = np.zeros(len(connected_to)).astype(float)

    def __call__(self,):
        return self.position, self.rotations, self.force, self.torque

##############################################################
def init_particles(d, angle, n_p = 2):
    """
    Initializes the system. For each n-particle system, the system is already 'pre-initialized'.
    A dictionary is returned in the form of:
    dict = {
        "p1": particle_class_1,
        "p2": particle_class_2,
        ...
        "pn": particle_class_n,
    }
    """
    # initialize particles, this can also be automated through a function
    if n_p == 2:
        r = d / 2
        particle1 = Particle(position = position(r, angle),connected_to = ["p2"])
        particle2 = Particle(position = position(r, angle + math.pi), connected_to = ["p1"])
        particles = [particle1, particle2]
    if n_p == 3:
        r = d / math.sqrt(3) # to keep the initial distance between connected particles constant
        particle1 = Particle(position = position(r, angle),connected_to = ["p2", "p3"])
        particle2 = Particle(position = position(r, angle + math.pi * 2/3), connected_to = ["p1", "p3"])
        particle3 = Particle(position = position(r, angle - math.pi * 2/3), connected_to = ["p1", "p2"])
        particles = [particle1, particle2, particle3]
    if n_p == 4:
        r = d / math.sqrt(2)
        particle1 = Particle(position = position(r, angle),connected_to = ["p4", "p2"])
        particle2 = Particle(position = position(r, angle + math.pi * 1/2), connected_to = ["p1", "p3"])
        particle3 = Particle(position = position(r, angle + math.pi * 2/2), connected_to = ["p2", "p4"])
        particle4 = Particle(position = position(r, angle + math.pi * 3/2), connected_to = ["p3", "p1"])
        particles = [particle1, particle2, particle3, particle4]
    if n_p == "4+":
        r = d / math.sqrt(2)
        particle1 = Particle(position = position(r, angle),connected_to = ["p4", "p2", "p3"])
        particle2 = Particle(position = position(r, angle + math.pi * 1/2), connected_to = ["p1", "p3", "p4"])
        particle3 = Particle(position = position(r, angle + math.pi * 2/2), connected_to = ["p2", "p4", "p1"])
        particle4 = Particle(position = position(r, angle + math.pi * 3/2), connected_to = ["p3", "p1", "p2"])
        particles = [particle1, particle2, particle3, particle4]
    if n_p == 5:
        r = d / (2 -2*math.cos(72*math.pi/180))**.5
        particle1 = Particle(position = position(r, angle),connected_to = ["p5", "p2"])
        particle2 = Particle(position = position(r, angle + math.pi * 2/5), connected_to = ["p1", "p3"])
        particle3 = Particle(position = position(r, angle + math.pi * 4/5), connected_to = ["p2", "p4"])
        particle4 = Particle(position = position(r, angle + math.pi * 6/5), connected_to = ["p3", "p5"])
        particle5 = Particle(position = position(r, angle + math.pi * 8/5), connected_to = ["p4","p1"])
        particles = [particle1, particle2, particle3, particle4, particle5]
    if n_p == "5+":
        r = d / (2 -2*math.cos(72*math.pi/180))**.5
        particle1 = Particle(position = position(r, angle),connected_to = ["p5", "p2", "p3", "p4"])
        particle2 = Particle(position = position(r, angle + math.pi * 2/5), connected_to = ["p1", "p3", "p4", "p5"])
        particle3 = Particle(position = position(r, angle + math.pi * 4/5), connected_to = ["p1", "p2", "p4", "p5"])
        particle4 = Particle(position = position(r, angle + math.pi * 6/5), connected_to = ["p1", "p2", "p3", "p5"])
        particle5 = Particle(position = position(r, angle + math.pi * 8/5), connected_to = ["p4","p1", "p2", "p3"])
        particles = [particle1, particle2, particle3, particle4, particle5]
    if n_p == 6:
        r = d / (2 -2*math.cos(45*math.pi/180))**.5
        particle1 = Particle(position = position(r, angle),connected_to = ["p6", "p2"])
        particle2 = Particle(position = position(r, angle + math.pi * 1/3), connected_to = ["p1", "p3"])
        particle3 = Particle(position = position(r, angle + math.pi * 2/3), connected_to = ["p2", "p4"])
        particle4 = Particle(position = position(r, angle + math.pi * 3/3), connected_to = ["p3", "p5"])
        particle5 = Particle(position = position(r, angle + math.pi * 4/3), connected_to = ["p4","p6"])
        particle6 = Particle(position = position(r, angle + math.pi * 5/3), connected_to = ["p5","p1"])
        particles = [particle1, particle2, particle3, particle4, particle5,particle6]
    if n_p == "6+":
        r = d / (2 -2*math.cos(45*math.pi/180))**.5
        particle1 = Particle(position = position(r, angle),connected_to = ["p6", "p2", "p3", "p4", "p5"])
        particle2 = Particle(position = position(r, angle + math.pi * 1/3), connected_to = ["p1", "p3", "p4", "p5", "p6"])
        particle3 = Particle(position = position(r, angle + math.pi * 2/3), connected_to = ["p2", "p4", "p5", "p6", "p1"])
        particle4 = Particle(position = position(r, angle + math.pi * 3/3), connected_to = ["p3", "p5", "p6", "p1", "p2"])
        particle5 = Particle(position = position(r, angle + math.pi * 4/3), connected_to = ["p4","p6", "p1", "p2", "p3"])
        particle6 = Particle(position = position(r, angle + math.pi * 5/3), connected_to = ["p5","p1", "p2", "p3", "p4"])
        particles = [particle1, particle2, particle3, particle4, particle5,particle6]
    if n_p == 9:
        Rt = d
        particle1 = Particle(position = np.array([-Rt, Rt]),connected_to = ["p2", "p4"])
        particle2 = Particle(position = np.array([0, Rt]), connected_to = ["p1", "p3", "p5"])
        particle3 = Particle(position = np.array([Rt, Rt]), connected_to = ["p2", "p6"])
        particle4 = Particle(position = np.array([-Rt, 0]), connected_to = ["p1", "p5", "p7"])
        particle5 = Particle(position = np.array([0, 0]).astype(float), connected_to = ["p2", "p4", "p6", "p8"])
        particle6 = Particle(position = np.array([Rt, 0]), connected_to = ["p3", "p5", "p9"])
        particle7 = Particle(position = np.array([-Rt, -Rt]), connected_to = ["p4", "p8"])
        particle8 = Particle(position = np.array([0, -Rt]), connected_to = ["p5", "p7", "p9"])
        particle9 = Particle(position = np.array([Rt,-Rt]), connected_to = ["p6", "p8"])
        particles = [particle1, particle2, particle3, particle4, particle5, particle6, particle7, particle8, particle9]
    if n_p == "9+":
        Rt = d
        particle1 = Particle(position = np.array([-Rt, Rt]),connected_to = ["p2", "p4", "p5"])
        particle2 = Particle(position = np.array([0, Rt]), connected_to = ["p1", "p3", "p5", "p4", "p6"])
        particle3 = Particle(position = np.array([Rt, Rt]), connected_to = ["p2", "p6", "p5"])
        particle4 = Particle(position = np.array([-Rt, 0]), connected_to = ["p1", "p5", "p7", "p8", "p2"])
        particle5 = Particle(position = np.array([0, 0]).astype(float), connected_to = ["p2", "p4", "p6", "p8", "p1", "p3", "p7", "p9"])
        particle6 = Particle(position = np.array([Rt, 0]), connected_to = ["p3", "p5", "p9", "p2", "p8"])
        particle7 = Particle(position = np.array([-Rt, -Rt]), connected_to = ["p4", "p8", "p5"])
        particle8 = Particle(position = np.array([0, -Rt]), connected_to = ["p5", "p7", "p9", "p4", "p6"])
        particle9 = Particle(position = np.array([Rt,-Rt]), connected_to = ["p6", "p8", "p5"])
        particles = [particle1, particle2, particle3, particle4, particle5, particle6, particle7, particle8, particle9]

    # create a dictionary holding all the particles
    names = [f"p{i+1}" for i in range(len(particles))]
    particles = dict(zip(names, particles))

    # compute for each particle all orientations that they are connected to, and save their initial distance
    for key in particles.keys():
        rotations = np.zeros([len(particles[key].connected_to)])
        rest_lengths = np.zeros([len(particles[key].connected_to)])
        for i, p_other in enumerate(particles[key].connected_to,0):
            # compute initial angle
            rotations[i] = compute_angle(particles[key].position, particles[p_other].position) 

            # compute initial distance, which is also the rest length of the "spring" that connects them
            rest_lengths[i] = math.sqrt(sum((particles[key].position - particles[p_other].position)**2))
        if len(rotations) > 0:
            particles[key].rotations = rotations
            particles[key].rest_lengths = rest_lengths

    # set center of mass at zero coordinate if not already so
    y_cor = -sum([p.position[1] for p in particles.values()])/len(particles)
    x_cor = -sum([p.position[0] for p in particles.values()])/len(particles)
    for key in particles.keys():
        particles[key].position[0] += x_cor
        particles[key].position[1] += y_cor
    return particles

##############################################################
def add_spin(p, particles, particles_old, RK = True):
    """
    tanh() Is limited by [-pi, +pi]. To keep track of the number of rotations, 
    this functions checks if this 'jump' from -pi to +pi occurred or vice-versa, 
    relative to the previous positions.
    """
    n_spins = np.zeros(len(particles[p].connected_to)).copy()
    for i,p_other in enumerate(particles[p].connected_to,0):
        # computes both angles before and after updating the position
        angle_bef = compute_angle(particles_old[p].position, particles_old[p_other].position)
        angle_now = compute_angle(particles[p].position, particles[p_other].position)
        # we don't want to register a switch from -0 to 0, only from -pi to +pi. Assuming the steps are small enough, this only registers the correct 'switches'
        if angle_now > math.pi/2 and angle_bef < - math.pi/2:   
            n_spins[i] = -1
        elif angle_now < -math.pi/2 and angle_bef > math.pi/2: 
            n_spins[i] = 1
    return n_spins    

def compute_angle(x1,x2):
    """
    Computes the angle of particle 1 relative to particle 2
    """
    r_vec = x1 - x2
    return np.arctan2(r_vec[1], r_vec[0])

def position(r, angle):
    """ 
    Transforms polar coordinates to cartesian coordinates
    """
    return r * np.array([math.cos(angle), math.sin(angle)])

##############################################################
def simulate(particles, t, plot = False, verbose = 0):
    """
    Simulates a given system of particles for a given period t.
    Faxen's laws are used to compute the velocity of each particle, 
    then the trajectory using Runge-Kutta 4th-order method.
    Each iteration an if-statement checks if a spring is extended past a certain threshold. 
    If so, this time is returned as the break up time.
    """
    # store results in dataframe
    # each line in the dataframe is [t, x1_x, x1_y, x1_rot, ... , xn_x, xn_y, xn_rot]
    results = np.zeros((1,1 + len(particles)*3)) # time + x,y,rotation
    for i,key in enumerate(particles.keys(),0):
        results[0,1+i*3] = particles[key].position[0]
        results[0,2+i*3] = particles[key].position[1]
        results[0,3+i*3] = particles[key].rotations[0]
    
    #######################
    if plot == 2:
        # n_particles = int((results.shape[1]-1)/3)
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()
    ######################

    # simulate particles
    TIME = 0
    i = 0
    while TIME < t:
        i += 1
        TIME += DT_P

        newrow = np.zeros((1, 1 + 3 * len(particles))).astype(float)
        newrow[0,0] = TIME

        # current holds for each particle the velocity and rotational velocity
        particles_old = dict(zip(particles.keys(), [Particle(other = particles[key]) for key in particles.keys()]))
        U, W = runge_kutta(particles)

        for j,key in enumerate(particles.keys(),0):
            # update position
            particles[key].position[0] = particles_old[key].position[0] + U[key][0] * DT_P
            particles[key].position[1] = particles_old[key].position[1] + U[key][1] * DT_P
            particles[key].rotations = particles_old[key].rotations + W[key] * DT_P

            # update dataframe
            newrow[0,1+j*3] = particles[key].position[0]
            newrow[0,2+j*3] = particles[key].position[1]
            newrow[0,3+j*3] = particles[key].rotations[0]

        for key in particles.keys():
            # check if rotation occured, i.e. if tangens function 'switched' from +pi to -pi or vice versa
            particles[key].n_spins = particles_old[key].n_spins + add_spin(key, particles, particles_old, RK = False)

        # check for break up and add results to dataframe
        broken = is_broken(particles, TIME, verbose)
        if broken:
            break
        results = np.vstack([results, newrow])

        # after the simulations, plot or animate rsults
        # if animate, update plot
        if plot == 2:
            pairs_done = []
            for j, p in enumerate(particles,0):
                plt.scatter(results[i,1+j*3], results[i,2+j*3], label = f"p{j+1}", s = 2000, color = "b")
                # plt.plot([results[i, 1+j*3], results[i, 1+j*3] + math.cos(results[i, 3+j*3]) *1e-5], [results[i, 2+j*3], results[i, 2+j*3]+ math.sin(results[i, 3+j*3]) *1e-5])
                for p_other in particles[p].connected_to:
                    if f"{p}{p_other}" not in pairs_done and f"{p_other}{p}" not in pairs_done:
                        pairs_done.append(f"{p}{p_other}")
                        pos1 = particles[p].position
                        pos2 = particles[p_other].position
                        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color = "b")
            limit = 3.5e-5
            plt.xlim(-limit,limit)
            plt.ylim(-limit,limit)
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.draw()
            plt.pause(0.0000001)
            plt.clf()
            fig.canvas.draw()
            # after completing one round, check for break up

    if plot == 1:
        t = pd.DataFrame(results)
        plot_results(results)
    if plot == 3:
        animate_results(results)

    if broken:
        return TIME
    # if not broken, None is returned

#############################################################
def is_broken(particles, TIME, verbose):
    """
    Checks if a break up occurred. The distance of all connections is computed and then compared to their initial distance.
    If so, the connection is removed and True is returned.
    """
    for key in particles.keys():
        for z,p_other in enumerate(particles[key].connected_to):
            stretched = math.sqrt(sum((particles[key].position - particles[p_other].position)**2))
            if stretched / particles[key].rest_lengths[z] > BREAK_UP_RATIO:
                if verbose > 2:
                    print(f"[{round(TIME,7)}] - {key}{p_other} broken up")

                if STOP_FIRST:
                    # for first break up no need to remove connection from system as we immediately quit
                    return True
                else:
                    # remove connection from main particle
                    particles[key].rotations = np.delete(particles[key].rotations,z)
                    particles[key].n_spins = np.delete(particles[key].n_spins, z)
                    particles[key].rest_lengths = np.delete(particles[key].rest_lengths, z)
                    particles[key].connected_to.remove(p_other)

                    # do the same for the other particle
                    ind = particles[p_other].connected_to.index(key)
                    particles[p_other].rotations = np.delete(particles[p_other].rotations, ind)
                    particles[p_other].n_spins = np.delete(particles[p_other].n_spins , ind)
                    particles[p_other].rest_lengths = np.delete(particles[p_other].rest_lengths, ind)
                    particles[p_other].connected_to.remove(key)

                    # if one particle broke up all its connections, i.e. is isolated, then return True
                    if len(particles[key].rotations) == 0 or len(particles[p_other].rotations) == 0:
                        return True
    return False

def runge_kutta(particles):
    """
    Computes velocity and rotation through Runge-Kutta 4th order method.
    Returns results in dictionary with the names of the particles as keys.
    The final dictionary is returned as:
    U = {
        "p1": U_1,
        ...
        "pn": U_n,
    }
    W = {
        "p1": W_1,
        ...
        "pn": W_n, 
    }
    """
    # we don't want to edit the existing dictionary
    # so two copies are made, store the beginning values
    particles_store = dict(zip(particles.keys(), [Particle(other = particles[key]) for key in particles.keys()]))
    particles_temp = dict(zip(particles.keys(), [Particle(other = particles[key]) for key in particles.keys()]))

    # dictionaries to store velocities
    U = dict(zip(particles.keys(),[[0,0,0,0] for p in particles.keys()]))
    W = dict(zip(particles.keys(),[[0,0,0,0] for p in particles.keys()]))

    # RK 1
    U_dict, W_dict = compute_derivative_all(particles_temp)
    for p in particles_temp:
        U_k = DT_P * U_dict[p].reshape(-1)
        W_k = DT_P * W_dict[p]
        U[p][0] = U_k
        W[p][0] = W_k
    # RK 2
    for p in particles_temp:
        particles_temp[p].position = particles_store[p].position + .5 * U[p][0]
        particles_temp[p].rotations = particles_store[p].rotations + .5 * W[p][0]
        particles_temp[p].n_spins = particles_store[p].n_spins + add_spin(p, particles_temp, particles_store)
    U_dict, W_dict = compute_derivative_all(particles_temp)
    for p in particles_temp:
        U_k = DT_P * U_dict[p].reshape(-1)
        W_k = DT_P * W_dict[p]
        U[p][1] = U_k
        W[p][1] = W_k
    # RK 3
    for p in particles:
        particles_temp[p].position = particles_store[p].position + .5 * U[p][1]
        particles_temp[p].rotations = particles_store[p].rotations + .5 * W[p][1]
        particles_temp[p].n_spins = particles_store[p].n_spins + add_spin(p, particles_temp, particles_store)
    U_dict, W_dict = compute_derivative_all(particles_temp)
    for p in particles_temp:
        U_k = DT_P * U_dict[p].reshape(-1)
        W_k = DT_P * W_dict[p]
        U[p][2] = U_k
        W[p][2] = W_k
    # RK 4
    for p in particles:
        particles_temp[p].position = particles_store[p].position + 1 * U[p][2]
        particles_temp[p].rotations = particles_store[p].rotations + 1 * W[p][2]
        particles_temp[p].n_spins = particles_store[p].n_spins + add_spin(p, particles_temp, particles_store)
    U_dict, W_dict = compute_derivative_all(particles_temp)
    for p in particles_temp:
        U_k = DT_P * U_dict[p].reshape(-1)
        W_k = DT_P * W_dict[p]
        U[p][3] = U_k
        W[p][3] = W_k

    # compute average of results
    weights = [1/6, 1/3, 1/3, 1/6]
    for p in particles:
        U[p] = sum([val*w for val,w in zip(U[p], weights)]) / DT_P
        W[p] = sum([val*w for val,w in zip(W[p], weights)])[0] / DT_P
    return U, W

def compute_derivative_all(particles):
    """
    Computes the velocity and rotation for all particles,
    and returns the results in a dictionary with 'p{i}' as keys
    """
    U = {}
    W = {}
    for p in particles:
        u, w = compute_derivative(p, particles)
        U[p] = u 
        W[p] = w
    return U, W

def compute_derivative(p, particles):
    """
    Computes the velocity and rotation for one particles,
    and returns U, W (U = np.ndarray, W = scalar)
    """
    # first compute total torques and forces on each particle
    forces = [compute_tot_force(p, particles) for p in particles.keys()]
    res = [compute_tot_torque(p, particles) for p in particles.keys()]
    torque = [r[0] for r in res]
    forces = [f + r[1] for f,r in zip(forces, res)]

    # the bending also applies a force on the particles: T1 + T2 in the direction of theta + pi/2
    for i, key in enumerate(particles.keys(),0):
        particles[key].force = forces[i]
        particles[key].torque = torque[i]
    
    # the external velocities by the shear flow
    U = np.array([SHEAR_RATE * particles[p].position[1],0]).reshape(-1,1)
    W = np.array([-SHEAR_RATE/2])

    # particle particle interactions
    for key in particles.keys():
        if key == p:
            U += 1/(6 * math.pi * VISCOSITY * A) * particles[p].force
            W += 1/(8 * math.pi * VISCOSITY * A**3) * particles[p].torque
        if key != p:
            U_acc, W_acc = compute_pp(particles[p], particles[key])
            U += U_acc
            W += W_acc
    return U, W

def compute_pp(p, p_other):
    """
    Computes the disturbance velocity of p_other on p
    """
    x1, w1, f1, t1 = p()
    x2, w2, f2, t2 = p_other()

    r_vec = (x1 - x2).reshape(-1,1)
    r = math.sqrt(sum(r_vec**2))
    rr = r_vec @ r_vec.T / r**3

    # compute disturbance velocities and rotation (caused by force / rate of strain / torque)
    U2_acc =      1/(8*math.pi*VISCOSITY) * (np.identity(2)/r + rr) @ f2 \
                    + -5/2 * A**3 * r_vec[1] * r_vec[0] * r_vec.copy().reshape(-1,1) / r**5 \
                    + 1/(8*math.pi*VISCOSITY*r**3) * t2 * np.array([-r_vec[1], r_vec[0]]).reshape(-1,1)# torque correction
    W2_acc = 1/(8*math.pi*VISCOSITY * r**3) * (r_vec[1] * f2[0] - r_vec[0] * f2[1]) \
                + -1/(16*math.pi*VISCOSITY * r**3) * t2
    return U2_acc, W2_acc

###############################################################
def compute_tot_force(p, particles):
    """
    Computes the total force on specified particle in the system. The force only comes from its connections.
    Therefore we iterate through all connections sum up the force.
    """
    tot_force = np.zeros((2,1)).astype(float)
    for i,p_other in enumerate(particles[p].connected_to,0):
        tot_force += compute_force(particles[p], particles[p_other], particles[p].rest_lengths[i])
    return tot_force

def compute_force(p, p_other, r0):
    """
    Computes the force that a spring (between p and p_other) applies on particle p
    """
    x1 = p.position
    x2 = p_other.position
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)

    r = math.sqrt(sum((x1 - x2)**2))
    r_vec = (x2 - x1)
    cor = max(1, (r0 - RMIN) / (r - RMIN))
    force = cor * (r - r0) * K * r_vec / r
    return force 

###############################################################
def compute_tot_torque(p, particles):
    """
    Computes the total torque on each particles, also returns the force that the torque causes 
    """
    tot_torque = 0
    tot_force = np.zeros((2,1)).astype(float)
    for p_other in particles[p].connected_to:
        res = compute_torque(particles[p], particles[p_other], p, p_other)
        tot_torque += res[0]
        tot_force += res[1]
    return tot_torque, tot_force

def compute_torque(p, p_other, key1, key2):
    x1, w1, x, x = p()
    x2, w2, x, x = p_other()
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    r = math.sqrt(sum((x1 -x2)**2))
    w1 = dict(zip(p.connected_to,w1))[key2]
    w2 = dict(zip(p_other.connected_to,w2))[key1]

    THETA_BEAM1 = compute_angle(x1,x2)[0]
    THETA_BEAM1 += dict(zip(p.connected_to,p.n_spins))[key2] * 2 * math.pi
    THETA_BEAM2 = compute_angle(x2,x1)[0]
    THETA_BEAM2 += dict(zip(p_other.connected_to,p_other.n_spins))[key1] * 2 * math.pi
    w1_diff =  THETA_BEAM1 - w1
    w2_diff =  THETA_BEAM2 - w2

    torque = 2 * K_BEND / r * (2*w1_diff + w2_diff)
    torque2 = 2 * K_BEND / r * (2*w2_diff + w1_diff)
    force = (torque + torque2)/r * np.array([math.cos(THETA_BEAM1 - math.pi/2), math.sin(THETA_BEAM1 - math.pi/2)]).reshape(-1,1)
    return torque, force

def plot_results(results):
    """
    Plot the trajectories of the particles. 
    Plot 1: trajectory of the position (x,y) of each particle.
    Plot 2: plot of the distance between particles over time
    Plot 3: the rotations for all particle over time
    Plot 4: the x and y position of one particle against time, to check for convergence at a certain time
    """
    n_particles = int((results.shape[1]-1)/3)
    fig, axs = plt.subplots(4,1, figsize = (7,7))

    # PLOT ONE: MOVEMENTS
    for i in range(n_particles):
        axs[0].plot(results[:,1+i*3], results[:,2+i*3], label = f"p{i+1}")

    # PLOT TWO: DISTANCE OVER TIME
    # first compute radius
    results_radius = np.zeros((results.shape[0], int(math.factorial(n_particles)/2)))
    done = []
    k = 0
    for i in range(n_particles):
        for j in range(n_particles):
            if i != j and f"{i}{j}" not in done and f"{j}{i}" not in done:
                done.append(f"{i}{j}")
                x1 = results[:,1+3*i]
                y1 = results[:,2+3*i]
                x2 = results[:,1+3*j]
                y2 = results[:,2+3*j]
                radius_ij = ((x1-x2)**2 + (y1-y2)**2)**.5
                results_radius[:,k] = radius_ij
                axs[1].plot(results[:,0], results_radius[:,k], label = f"p{i+1}p{j+1}")
                k+=1

    # PLOT THREE: ROTATION OVER TIME
    for i in range(n_particles):
        axs[2].plot(results[:,0], results[:,3+3*i], label = f"p{i+1}")

    # PLOT FOUR: ONE PARTICLE OVER TIME
    plot_p_n = 1 # take the first particle
    axs[3].plot(results[:,0], results[:,1+3*plot_p_n])
    axs[3].plot(results[:,0], results[:,2+3*plot_p_n])

    for ax in axs:
        ax.legend()
        ax.grid()
    plt.show()

def animate_results(results):
    """
    This function creates a GIF of the positions of all particles. At the moment it is not able to plot cross connections.
    """
    results = results
    fig = plt.figure()
    ax = plt.axes(xlim=(-2e-5, 2e-5), ylim=(-2e-5, 2e-5))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    line, = ax.plot([],[], 'o-', lw = 2, ms = 60,)
    
    def init():
        line.set_data([],[])
        return line
    
    def animate(i):
        l = results[i]
        # get data
        n_p = int((len(l)-1)/3)
        if n_p == 2:
            x = l[[1+j*3 for j in range(n_p)]]
            y = l[[2+j*3 for j in range(n_p)]]
        else:
            index1 = [1+j*3 for j in range(n_p)]
            index2 = [2+j*3 for j in range(n_p)]
            index1.append(1)
            index2.append(2)
            x = l[index1]
            y = l[index2]

        # plot data
        line.set_data(x,y)

        return line
    # anim = animation.FuncAnimation(fig, animate, init_func = init, frames = results.shape[0], interval = .01, blit = True)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = results.shape[0], interval = DT_P*1000, blit = False)
    anim.save(f"animations/n_p={N_P}_sr={SHEAR_RATE}.gif", fps = 30)
    # plt.show()

def fit_power_function(x,y):
    def log(series):
        return np.log(series)
    df = pd.DataFrame({"x":x, "y":y})
    df = df.dropna()
    # for column in df.columns:
    #     df[f"log_{column}"] = np.log(df[column])
    mod = smf.ols(formula = 'log(y) ~ log(x)', data = df)
    res = mod.fit()
    print(res.summary())

def find_critical_shear_rate(system = 4, sr_min = 3, sr_max = 6, precision = .1, non_dim = False, verbose = 0):
    """
    Determines the critical shear rate for a break up to occur. 
    A certain starting range is defined after which that is reduced up to a certain accuracy using bisection method.
    """
    global DT_P
    global SHEAR_RATE
    while sr_max - sr_min > precision:
        SHEAR_RATE = (sr_max + sr_min)/2

        DT_P = DT/SHEAR_RATE
            # average out the break up time in all orientations
        angle = 0
        particles = init_particles(d = D0, angle = angle, n_p = system)

        # simulate particles, returns (time of break up, particle1, particle2) that broke up
        r = simulate(particles, t = T/SHEAR_RATE, plot = 0) # for 3; animation, save is off
        if r:
            sr_max = SHEAR_RATE
        else:
            sr_min = SHEAR_RATE
        if verbose > 0:
            print(sr_min, sr_max)
    if non_dim:
        return sr_min * VISCOSITY * A / K, sr_max * VISCOSITY * A / K
    else:
        return sr_min, sr_max

def find_all_critical_shear_rates(systems = [2,3,4,5,6], sr_min = 5, sr_max = 5, precision = .1, non_dim = False, verbose = 0, test = True):
    """
    Determine minimum shear rate for a list of given systems, plot and save.
    """
    fig, axs = plt.subplots(1,1, figsize = (6,6))
    vals = []
    min_vals = []
    max_vals = []
    for system in systems:
        if verbose > 0:
            print(system)
        val_min, val_max = find_critical_shear_rate(system = system, sr_min = sr_min, sr_max = sr_max, precision = precision, non_dim = non_dim, verbose = verbose)
        val = (val_min + val_max)/2
        vals.append(val)
        min_vals.append(val - val_min)
        max_vals.append(val_max- val)
    systems = [2,3,4,5,6]
    axs.errorbar(systems, vals, yerr = [min_vals, max_vals])
    plt.xlabel("N-particle System")
    plt.ylabel("Minimum Shear Rate for Break Up [$s^{-1}$]")
    plt.title("Minimum Shear Rate for N-particle Systems")
    plt.xlim(2,6)
    plt.xticks([2,3,4,5,6])
    if not test:
        plt.savefig(f'min_sr/min_sr_{"".join([f"{el}" for el in systems])}_non-dim={non_dim}.png', dpi = 400)
    plt.show()

def simulate_all_shear(systems = [5, "5+"], shear_rates = np.arange(5, 25, .25), angles = np.arange(0, 180, 10), non_dim = False, verbose = 0, test = False):
    """
    Vary the shear rate for a list of systems, and plot the break up times for each system.
    The results are plotted on a logarithmic scale, and have the option to be transformed to a non-dimensional form.
    The break up time is multiplied by the shear rate, and the shear rate on the x-axis is multiplied by (mu*radius/k_spr)
    """
    global DT_P
    global SHEAR_RATE

    # for particles ranging between 2 and 5
    for n_p in systems:
        # execute functions
        res = []
        if verbose >= 0:
            print(f"System: {n_p}")
        for SHEAR_RATE in shear_rates:
            DT_P = DT/SHEAR_RATE

            results = []
            # average out the break up time in all orientations
            for angle in angles:
                # initialize particles 
                angle = angle/180 * math.pi
                particles = init_particles(d = D0, angle = angle, n_p = n_p)

                # simulate particles, returns (time of break up, particle1, particle2) that broke up
                r = simulate(particles, t = T/SHEAR_RATE, plot = 0, verbose = verbose) # for 3; animation, save is off
                if r:
                    if non_dim:
                        results.append(r * SHEAR_RATE)
                    else:
                        results.append(r)
                else:
                    results.append(float("NaN"))  
                    if verbose > 1:
                        # if the threshold for a break up is met, all orientations will result in a break up, only at different times, i.e. if one simulation does not break up, we can skip all other orientations
                        print(f"[{SHEAR_RATE}] - Early stop at {round(angle * 180 / math.pi)}")
                    break
            if verbose > 0:
                print(f"system: {n_p} | shear rate: {SHEAR_RATE} | avg. break-up: {np.nanmean(results)} | completed {len(results)}/{len(angles)}")
            if np.nanmean(results) == float("NaN"):
                res.append(None)
            else:res.append(np.nanmean(results))

        if non_dim:
            shear_rates1 = shear_rates.copy() * VISCOSITY * A / K
        else:
            shear_rates1 = shear_rates.copy()
        fit_power_function(shear_rates1, res)
        plt.plot(shear_rates1, res, label = f"{n_p}")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.title("Simulation of Different Systems")
    if non_dim:
        plt.xlabel("$\mu\dot{\gamma}a/k_{spr}$ []")
        plt.ylabel("$\dot{\gamma} \cdot t_{br}$ []")
    else:
        plt.xlabel("Shear Rate [$s^{-1}$]")
        plt.ylabel("Time of Break Up [s]")
    if not test:
        plt.savefig(f'vary_sr/sim_{"".join([f"{el}" for el in systems])}_{int(min(shear_rates))}-{int(max(shear_rates))}_non-dim={non_dim}_stop-first={STOP_FIRST}.png', dpi = 400)
        plt.show()

def simulate_all_distance(systems = [5, "5+"], initial_distances = np.arange(2,20,2), angles = np.arange(0, 180, 10), non_dim = False, verbose = 0, test = False):
    """
    Plot break up time for varied initial distance for a list of systems
    """
    global DT_P
    global SHEAR_RATE
    global D0

    # for particles ranging between 2 and 5
    for n_p in systems:
        # execute functions
        res = []
        if verbose >= 0:
            print(f"System: {n_p}")
        for D0 in initial_distances:
            DT_P = DT/SHEAR_RATE
            D0 *= A

            results = []
            # average out the break up time in all orientations
            for angle in angles:
                # initialize particles 
                angle = angle/180 * math.pi
                particles = init_particles(d = D0, angle = angle, n_p = n_p)

                # simulate particles, returns (time of break up, particle1, particle2) that broke up
                r = simulate(particles, t = T/SHEAR_RATE, plot = 0, verbose = verbose) # for 3; animation, save is off
                if r:
                    if non_dim:
                        results.append(r * SHEAR_RATE)
                    else:
                        results.append(r)
                else:
                    results.append(float("NaN"))  
                    if verbose > 1:
                        # if the threshold for a break up is met, all orientations will result in a break up, only at different times, i.e. if one simulation does not break up, we can skip all other orientations
                        print(f"[{SHEAR_RATE}] - Early stop at {round(angle * 180 / math.pi)}")
                    break
            if verbose > 0:
                print(f"system: {n_p} | distance: {round(D0,6)} | shear rate: {SHEAR_RATE} | avg. break-up: {np.nanmean(results)} | completed {len(results)}/{len(angles)}")
            if np.nanmean(results) == float("NaN"):
                res.append(None)
            else:res.append(np.nanmean(results))

        if non_dim:
            initial_distances1 = initial_distances.copy() * VISCOSITY * A / K
        else:
            initial_distances1 = initial_distances.copy()
        fit_power_function(initial_distances1, res)
        plt.plot(initial_distances1, res, label = f"{n_p}")
    plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    plt.title("Simulation of Different Systems")
    if non_dim:
        plt.xlabel("$\mu\dot{\gamma}a/k_{spr}$ []")
        plt.ylabel("$\dot{\gamma} \cdot t_{br}$ []")
    else:
        plt.xlabel("Distance / Radius []")
        plt.ylabel("Time of Break Up [s]")
    if not test:
        print(SHEAR_RATE*VISCOSITY * A / K)
        plt.savefig(f'vary_d/sim_{"".join([f"{el}" for el in systems])}_sr={SHEAR_RATE}_distances={int(min(initial_distances))}-{int(max(initial_distances))}_non-dim={non_dim}_stop-first={STOP_FIRST}.png', dpi = 400)
    plt.show()


def within_range(val, min = 0, max = 50):
    if min <= val <= max:
        return True
    return False

def ask_input(targets, question, crit = None, min = None, max = None):
    """
    Function that asks to give input from a list of allowed values. If not valid, the questions is repeated.
    """
    val = False
    while not val:
        if question:
            option = input(f'{question}: {"/".join([str(t) for t in targets])} - ').lower()
        else:
            option = input(f'options: {"/".join([str(t) for t in targets])} - ').lower()
        if crit:
            if within_range(float(option), min, max):
                val = True
        else:
            if option in targets:
                val = True
    return option

# fixed parameters and variables
VISCOSITY = 1e-3 # viscosity of water
K = 1e-6#5e-7 # 1e-6 should be fine, now used 20x smaller
A = 5e-6 
K_BEND = K * math.pi * A**3 /4 # times something to compensate for something
RMIN = 2 * A # minimum distance that the particles can approach each other
DT = .1 # 0.1
T = 20 # the simulation time

D0 = 3 * A # 3 between 1 and 5 radii. This is also the initial distance between the particles so the spring starts at rest
BREAK_UP_RATIO = 1.2 # between 10 and 50 %
STOP_FIRST = False # stop at break up of a first connection, or stop when one particle lost all its connections

if __name__ == "__main__":
    # ask type of simulation and parameters
    option = ask_input(targets = ["single", "vary_sr", "vary_d", "min_d"], question = "Type of simulation to perform")
    if option != "single":
        SYSTEMS = [2,3,"4+","5+","6+"] if ask_input(["true", "false"], question = "Cross connections allowed") == "true" else [2,3,4,5,6]
        test = True if ask_input(["true","false"], "Test simulation? I.e. safe figure") == "true" else False
        non_dim = True if ask_input(["true","false"], "Transform results to non-dimensional form?") == "true" else False
    else:
        N_P = ask_input(["2","3","4","5","6", "4+", "5+", "6+","9+","9"] , "What system to simulate?")
        try:N_P = int(N_P)
        except: pass
        plot = int(ask_input(["1","2","3"], "Plot/Animate/GIF"))
    if option == "single" or option == "vary_d":
        SHEAR_RATE = int(ask_input([0, 50], "What shear rate?", within_range, 0, 50))

    ######################## TESTING SINGLE SIMULATION ########################
    if option == "single":
        DT_P = DT / SHEAR_RATE
        particles = init_particles(d = D0,angle = 0/180 * math.pi, n_p = N_P) # 50
    
        # simulate particles, returns (time of break up, particle1, particle2) that broke up
        r = simulate(particles, t = T/SHEAR_RATE, plot = plot) # for 3; animation, save is off
        print(r)
    ######################## VARY SHEAR RATE FOR MULTIPLE SYSTEMS ########################
    elif option == "vary_sr":
        simulate_all_shear(
            systems = SYSTEMS,
            shear_rates = np.arange(4,25,.5), 
            angles = np.arange(0,180,5), 
            non_dim = non_dim,
            verbose = 1,
            test = test,
            )-1,
    ######################## VARY INIT DISTANCE FOR MULTIPLE SYSTEMS ########################
    elif option == "vary_d":
        simulate_all_distance(
            systems = SYSTEMS,
            initial_distances= np.arange(2.5,20,.5), 
            angles = np.arange(0,180,5), 
            non_dim = non_dim,
            verbose = 1,
            test = test,
            )-1,
    ######################## DETERMINE CRITICAL SHEAR RATE FOR MULTIPLE SYSTEMS ########################
    elif option == "min_d":
        # find_critical_shear_rate(system = 5, sr_min = 3, sr_max = 9, precision = 0.1, verbose = 1)
        find_all_critical_shear_rates(
            systems = SYSTEMS, 
            sr_min = 2.5, 
            sr_max = 25, 
            precision = .01, 
            verbose = 2, 
            non_dim = non_dim,
            test = test,)