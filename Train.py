import numpy as np
import os
import argparse
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Dataset')
    parser.add_argument('--train-label', type=str, default='Label/labels.txt')
    parser.add_argument('--train-target', type=str, default='HalfSquat')
    parser.add_argument('--output', type=str, default='Model')
    parser.add_argument('--scaler', type=str, default='Scaler')
    return parser.parse_args()

class BaseTrain:
    def __init__(self, args):
        self.Dataset = args.dataset
        self.TrainLabel = args.train_label
        self.TrainTarget = args.train_target
        self.OutputModel = args.output
        self.Scaler = args.scaler
        self.final_classes = ["HalfSquat", "KneeRaise", "ShoulderBladeStretch", "LateralRaise", "Others"]

    def load_data(self):
        distance_dir = os.path.join(self.Dataset, 'Distances')
        action_classes = os.listdir(distance_dir)
        Labels = []
        X = []
        y = []

        # main_actions = ["HalfSquat", "KneeRaise", "ShoulderBladeStretch", "LateralRaise"]
        main_actions = self.TrainTarget

        sequences = []
        seq_labels = []

        for action in action_classes:
            distance_action_dir = os.path.join(distance_dir, action, 'Distances')
            distance_files = sorted(os.listdir(distance_action_dir))
            final_label = action if action == main_actions else "Others"
            for dist_file in distance_files:
                dist_path = os.path.join(distance_action_dir, dist_file)
                distances = np.load(dist_path)  # shape: (timesteps, features)
                sequences.append(distances)
                seq_labels.append(final_label)

        # 找出最大序列長度
        max_length = max(seq.shape[0] for seq in sequences)

        # 假設所有序列的 feature_dim 都相同，取第一筆資料的 features 數量即可
        feature_dim = sequences[0].shape[1]

        padded_sequences = []
        for seq in sequences:
            if seq.shape[0] < max_length:
                pad_length = max_length - seq.shape[0]
                seq = np.vstack([seq, np.zeros((pad_length, feature_dim))])
            flattened = seq.reshape(-1)
            padded_sequences.append(flattened)

        X = np.array(padded_sequences)
        y = np.array(seq_labels)

        # 將 max_length 與 feature_dim 傳出，以便在 train 時存檔
        return X, y, seq_labels, max_length, feature_dim

    def train(self):
        X, y, Labels, max_length, feature_dim = self.load_data()

        with open(self.TrainLabel, 'w') as f:
            for label in sorted(set(y)):
                f.write(label + "\n")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if not os.path.exists(self.Scaler):
            os.makedirs(self.Scaler)
        joblib.dump(le, os.path.join(self.Scaler, f'{self.TrainTarget}LabelEncoder.pkl'))
        joblib.dump(scaler, os.path.join(self.Scaler, f'{self.TrainTarget}Scaler.pkl'))

        # 將 max_length 與 feature_dim 存檔
        joblib.dump((max_length, feature_dim), os.path.join(self.Scaler, f'{self.TrainTarget}SeqInfo.pkl'))

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        print(f"測試集準確率: {accuracy:.4f}")

        if not os.path.exists(self.OutputModel):
            os.makedirs(self.OutputModel)
        joblib.dump(knn, os.path.join(self.OutputModel, f'{self.TrainTarget}.pkl'))

        print(f'模型已儲存至 {self.OutputModel}')
        print(f'標準化器、標籤編碼器及序列資訊已儲存至 {self.Scaler}')

if __name__ == '__main__':
    args = get_parser()
    main = BaseTrain(args)
    main.train()
