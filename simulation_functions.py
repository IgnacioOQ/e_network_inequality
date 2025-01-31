from imports import *
from agents import BetaAgent
from model import Model
from network_utils import *
from network_randomization import *
from network_generation import *

G_default = barabasi_albert_directed(100,5)

def generate_parameters(_,G=G_default):

    unique_id =  uuid.uuid4().hex
    # I am not sure what the three lines below are for
    process_seed = int.from_bytes(os.urandom(4), byteorder='little')
    random.seed(process_seed)
    np.random.seed(process_seed)

    # Now what all simulations share
    uncertainty = random.uniform(.0007, .0025)
    n_experiments = random.randint(1, 15)

    # now we pick a random number
    p_rewiring = np.random.rand()*0.5

    # Do randomization
    randomized_network = randomize_network(G, p_rewiring=p_rewiring)

    params = {
        'randomized': True,
        "unique_id": unique_id,
        "n_agents": int(len(randomized_network.nodes)),
        "network": randomized_network,
        "uncertainty": float(uncertainty),
        "n_experiments": int(n_experiments),
        "p_rewiring": float(p_rewiring),
    }
    stats = network_statistics(randomized_network)
    for stat in stats.keys():
     params[stat] = stats[stat]

    return params

test = False
if test:
    params = generate_parameters('test',G_default)
    print(params)

def run_simulation_with_params(param_dict, number_of_steps=20000, show_bar=False):
    # Extract the network directly since it's already a NetworkX graph object
    my_network = param_dict['network']
    # Other parameters are directly extracted from the dictionary
    my_model = Model(my_network, n_experiments=param_dict['n_experiments'],
                     uncertainty=param_dict['uncertainty'])
    # Run the simulation with predefined steps and show_bar option

    my_model.run_simulation(number_of_steps=number_of_steps, show_bar=show_bar)
    result_dict = {
        key: value
        for key, value in param_dict.items()
        if isinstance(value, (int, float, str, tuple, list, bool))}

    result_dict['share_of_correct_agents_at_convergence'] = my_model.conclusion
    result_dict['convergence_step'] = my_model.n_steps # takes note of the last reported step

    return result_dict

# Wrapper function for multiprocessing
def run_simulation_wrapper(param_dict, number_of_steps=500):
    return run_simulation_with_params(param_dict, number_of_steps=number_of_steps, show_bar=False)