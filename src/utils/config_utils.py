import os
import configparser

def read_config(config_file=None):
   
    if config_file is None:
        # Construct the path to the config file relative to the project's root
        config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))

    config = configparser.ConfigParser()
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config.read(config_file)
    return config


def get_db_config(config):
    
    try:
        db_config = {
            'host': config['POSTGRESQL']['host'].strip(),
            'port': int(config['POSTGRESQL']['port'].strip()),
            'dbname': config['POSTGRESQL']['dbname'].strip(),
            'user': config['POSTGRESQL']['user'].strip(),
            'password': config['POSTGRESQL']['password'].strip(),
        }
        return db_config
    except KeyError as e:
        raise KeyError(f"Missing database configuration key: {e}")
