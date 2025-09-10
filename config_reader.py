import configparser
from os.path import exists
from numpy import array, vstack, zeros

def read_config(path):
    config = configparser.ConfigParser()
    if not exists(path):
        raise FileExistsError("The filename you provided does not exist, please check its path")
    else: 
        pass
    config.read(path)
    # Get parameters from each section of ini file
    Vessel = SectionMap(config, 'Vessel')
    Simulation = SectionMap(config, 'Simulation')
    Param = SectionMap(config, 'Parameters')
    Network = read_network(config)
    return Vessel, Simulation, Param, Network

def read_network(config):
    
    network = {}

    for node_str, line in config['Network'].items():
        node = int(node_str.strip())

        if '|' not in line:
            raise ValueError(f"Invalid line format for node {node}: '{line}'")

        # Split into parent and daughter parts
        parts = [part.strip() for part in line.split('|')]
        parents_part = parts[0].replace("parents:", "").strip()
        daughters_part = parts[1].replace("daughters:", "").strip()

        # Parse parents
        if parents_part.lower() == 'none' or parents_part == '':
            parents = []
        else:
            parents = [int(p.strip()) for p in parents_part.split(',') if p.strip()]

        # Parse daughters
        if daughters_part == '':
            daughters = []
        else:
            daughters = [int(d.strip()) for d in daughters_part.split(',') if d.strip()]

        network[node] = {
            'parents': parents,
            'daughters': daughters
        }

    return network

def SectionMap(config, section):
    list_opt = config.options(section)
    dict_section = {}
    for opt in list_opt:
        if opt in ["pressure_model", "anomalies", "inlet", "outlet", "path", "net"]: # Have to differentiate cases where arg is string
            dict_section[opt] = config.get(section, opt)
        elif opt in ["connec"]:
            value_list = config.get(section, opt).split('/')
            x = []
            for item in value_list:
                t = [int(value) for value in item.split(',')]
                x.append(t)
            dict_section[opt]= vstack(x)
        else:
            try:    # try, except is here to differentiate list of value and single value
                dict_section[opt] = config.getfloat(section, opt) 
            except ValueError:
                value_list = config.get(section, opt).split(',')
                dict_section[opt] = array([float(value) for value in value_list]) 
    return dict_section

