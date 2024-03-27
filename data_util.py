import os
import numpy as np
import tensorflow as tf


class MetaDataLoader:
    def __init__(self, data_dir, num_samples_per_location=100):
        self.data_dir = data_dir
        self.num_samples_per_location = num_samples_per_location

    def _load_and_process_data(self, locations):
        data_dict = {}

        for location in locations:
            location_dir = os.path.join(self.data_dir, location)

            train_data_path = os.path.join(location_dir, "train_data.npy")
            train_label_path = os.path.join(location_dir, "train_label.npy")
            vali_data_path = os.path.join(location_dir, "vali_data.npy")
            vali_label_path = os.path.join(location_dir, "vali_label.npy")

            # Load training data and labels
            train_data = np.load(train_data_path)
            train_label = np.load(train_label_path)

            # Load validation data and labels
            vali_data = np.load(vali_data_path)
            vali_label = np.load(vali_label_path)

            # Store data and labels in the data_dict with location as key
            data_dict[location] = {
                'train_data': train_data,
                'train_label': train_label,
                'vali_data': vali_data,
                'vali_label': vali_label
            }

        return data_dict

    def _create_episode(self, locations):

        data_dict = self._load_and_process_data(locations)

        # --------- Create the support set -------------
        selected_data = []
        selected_labels = []

        for location in locations:
            # Initialize temporary lists
            temp_data = []
            temp_labels = []

            # Iterate over samples in the location's training data
            for data, label in zip(data_dict[location]['train_data'], data_dict[location]['train_label']):
                if np.sum(label == 1) > 500:  # Check if label has more than 500 pixels of class 1
                    temp_data.append(data)
                    temp_labels.append(label)

            # Ensure you have enough samples
            if len(temp_data) >= self.num_samples_per_location:
                selected_data.extend(temp_data[:self.num_samples_per_location])
                selected_labels.extend(
                    temp_labels[:self.num_samples_per_location])
            else:
                # Handle cases where not enough samples meet the criteria in the location
                print(
                    f"Warning: Not enough samples with >500 pixels of class 1 in location for support set in {location}")

        data = np.array(selected_data)
        labels = np.array(selected_labels)

        # Shuffle the data and labels if needed
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        support_set_data = data[indices]
        support_set_labels = labels[indices]

        # --------- End create the support set -------------

        # --------- Create the query set -------------
        selected_data = []
        selected_labels = []

        for location in locations:
            # Initialize temporary lists
            temp_data = []
            temp_labels = []

            # Iterate over samples in the location's training data
            for data, label in zip(data_dict[location]['vali_data'], data_dict[location]['vali_label']):
                if np.sum(label == 1) > 500:  # Check if label has more than 500 pixels of class 1
                    temp_data.append(data)
                    temp_labels.append(label)

            # Ensure you have enough samples
            if len(temp_data) >= self.num_samples_per_location:
                selected_data.extend(temp_data[:self.num_samples_per_location])
                selected_labels.extend(
                    temp_labels[:self.num_samples_per_location])
            else:
                # Handle cases where not enough samples meet the criteria in the location
                print(
                    f"Warning: Not enough samples with >500 pixels of class 1 in location for query set in {location}")

        data = np.array(selected_data)
        labels = np.array(selected_labels)

        # Shuffle the data and labels if needed
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        query_set_data = data[indices]
        query_set_labels = labels[indices]

        # --------- End create the query set -------------

        return support_set_data, support_set_labels, query_set_data, query_set_labels

    def create_multi_episodes(self, num_episodes, locations):
        episodes = []
        for _ in range(num_episodes):
            support_set_data, support_set_labels, query_set_data, query_set_labels = self._create_episode(
                locations)
            episode = {
                "support_set_data": support_set_data,
                "support_set_labels": support_set_labels,
                "query_set_data": query_set_data,
                "query_set_labels": query_set_labels
            }
            episodes.append(episode)
        return episodes
