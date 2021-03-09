import os

import numpy as np
import pandas as pd
import folium
DATA_PATH = '/home/csu2020/Zhuge/Traffic/Graph-WaveNet/data'
class Node():
    def __init__(self, id: int, locaton: list):
        self.id = id
        self.location = locaton

class Explore():
    def __init__(self, name='metr'):
        self.basemap = None # base map layer
        if name == 'metr':
            self.name = 'metr'
            self.data = os.path.join(DATA_PATH, 'sensor_graph', 'graph_sensor_locations.csv') # locations
            self.savepath = os.path.join('maps', self.name)
        elif name == 'pems-bay':
            self.name = 'pems-bay'
            self.data = os.path.join(DATA_PATH, 'sensor_graph', 'graph_sensor_locations_bay_uni.csv')  # locations
            self.savepath = os.path.join('maps', self.name)
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        self.node_list = []
        self.center_location = []
    def load_data(self):
        """
        load data and store them to Node datastruture
        :return:
        """
        print("*****load data***")
        self.data = pd.read_csv(self.data)
        import tqdm
        print(self.data.shape)
        print(self.data.head())
        data_list = []
        self.center_location = [np.mean(self.data["latitude"]), np.mean(self.data["longitude"])]
        for index, row in tqdm.tqdm(self.data.iterrows()):
            node = Node(id=row['sensor_id'], locaton=[row['latitude'], row['longitude']])
            data_list.append(node)

        return data_list


    def add_data_to_map(self, data_list=None):
        # test_nodes = [
        #     Node(123, [34.15497,-118.31829]),
        #     Node(345, [34.11621, -118.23799]),
        #     Node(7654634, [34.11641, -118.23819]),
        #     Node(345, [34.07248, -118.26772])
        # ]
        print("*****add_data_to_map***")
        if data_list is None:
            data_list = self.node_list
        for node in data_list:
            folium.Marker(node.location, popup=node.id).add_to(self.basemap)

    def run(self):

        res = self.load_data()
        self.init_basemap()
        self.add_data_to_map(res)
        self.basemap.save(os.path.join(self.savepath, 'index.html'))

    def init_basemap(self, center_loc=None):
        if center_loc is None and len(self.center_location)==2:
            center_loc = self.center_location
        assert len(center_loc) == 2
        self.basemap = folium.Map(location=center_loc)



if __name__ == "__main__":
    e = Explore(name='pems-bay')
    e.run()
    # data = pd.read_csv(os.path.join(DATA_PATH, 'sensor_graph', 'graph_sensor_locations_bay.csv'))
    # columns = ['sensor_id','latitude','longitude']
    # data.columns = columns
    # print(data.head())
    # data.to_csv(os.path.join(DATA_PATH, 'sensor_graph', 'graph_sensor_locations_bay_uni.csv'))