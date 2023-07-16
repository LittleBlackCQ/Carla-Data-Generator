import utils
from Data_Collector import DataCollector
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='tesla3.yaml', help='specify the config for collector')

    args = parser.parse_args()
    cfg_file = os.path.join("config", args.cfg_file)
    collector_config = utils.config_from_yaml(cfg_file)

    data_collector = DataCollector(collector_config)
    data_collector.start_collecting()
        
if __name__ == '__main__':
    main()
