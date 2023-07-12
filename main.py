import utils
from Data_Collector import DataCollector
import argparse

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='.\\config\\three_lidars.yaml', help='specify the config for collector')

    args = parser.parse_args()
    collector_config = utils.config_from_yaml(args.cfg_file)

    data_collector = DataCollector(collector_config)
    data_collector.start_collecting()
        
if __name__ == '__main__':
    main()
