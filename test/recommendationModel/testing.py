# testing.py

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data_collection import collect_user_interactions, collect_content_metadata, collect_contextual_info, feature_engineering
from feature_embedding import create_text_embeddings, create_categorical_embeddings, scale_numerical_features
from shallow_tower import build_shallow_tower
from multi_task_learning import build_multi_task_model, compile_multi_task_model
from training import train_model

class TestDataCollection(unittest.TestCase):
    def test_collect_user_interactions(self):
        data = collect_user_interactions()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('user_id', data.columns)

    def test_feature_engineering(self):
        interactions = collect_user_interactions()
        metadata = collect_content_metadata()
        context = collect_contextual_info()
        features = feature_engineering(interactions, metadata, context)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertFalse(features.empty)
        self.assertIn('hour_of_day', features.columns)

class TestFeatureEmbedding(unittest.TestCase):
    def test_create_text_embeddings(self):
        metadata = collect_content_metadata()
        embeddings = create_text_embeddings(metadata['title'])
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], len(metadata))

    def test_create_categorical_embeddings(self):
        features = pd.DataFrame({
            'category': ['Pets', 'Food', 'News', 'Gaming']
        })
        embeddings = create_categorical_embeddings(features, ['category'])
        self.assertIn('category', embeddings)
        self.assertIsInstance(embeddings['category'], np.ndarray)

class TestShallowTower(unittest.TestCase):
    def test_build_shallow_tower(self):
        input_shape = 128
        model = build_shallow_tower(input_shape)
        self.assertEqual(model.input_shape[1], input_shape)
        self.assertIsNotNone(model.output_shape)

class TestMultiTaskLearning(unittest.TestCase):
    def test_build_multi_task_model(self):
        input_shape = 32
        model = build_multi_task_model(input_shape)
        self.assertEqual(len(model.output_names), 3)
        self.assertIn('click_output', model.output_names)

class TestTraining(unittest.TestCase):
    @patch('training.train_model')
    def test_train_model(self, mock_train_model):
        features = pd.DataFrame(np.random.rand(100, 128))
        labels = {
            'click_output': np.random.randint(2, size=(100, 1)),
            'watch_time_output': np.random.rand(100, 1),
            'like_output': np.random.randint(2, size=(100, 1))
        }
        train_model(features, labels)
        mock_train_model.assert_called_with(features, labels)

if __name__ == '__main__':
    unittest.main()