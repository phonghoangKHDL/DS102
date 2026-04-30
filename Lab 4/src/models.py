import numpy as np

# CẤU TRÚC NODE CỦA CÂY
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature     # Chỉ số cột đặc trưng dùng để chia
        self.threshold = threshold # Giá trị ngưỡng để chia nhánh
        self.left = left           # Nhánh trái
        self.right = right         # Nhánh phải
        self.value = value         # Giá trị nhãn nếu node này là lá 

    def is_leaf_node(self):
        return self.value is not None


# DECISION TREE
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features 
        self.root = None

    def fit(self, X, y):
        n_feats = X.shape[1]
        self.n_features = n_feats if not self.n_features else min(self.n_features, n_feats)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_total_features = X.shape

        # Điều kiện dừng: Đạt độ sâu tối đa, ít hơn số mẫu tối thiểu, hoặc tất cả dữ liệu cùng 1 nhãn
        if (depth >= self.max_depth or n_samples < self.min_samples_split or np.all(y == y[0])):
            return Node(value=self._most_common_label(y))

        # Chọn ngẫu nhiên tập hợp các đặc trưng
        feat_idxs = np.random.choice(n_total_features, self.n_features, replace=False)

        # Tìm điểm chia tốt nhất
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        # Nếu không tìm được cách chia nào giúp tăng Information Gain
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        # Tạo node con
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        # Đề phòng trường hợp chia xong một nhánh bị rỗng
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=self._most_common_label(y))

        # Đệ quy mọc cây
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _information_gain(self, y, X_column, threshold):
        # Entropy cha
        parent_entropy = self._entropy(y)
        
        # Chia dữ liệu
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Entropy con
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information Gain
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)



# RANDOM FOREST
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # Lấy dự đoán từ tất cả các cây
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return self._majority_vote(tree_preds)
    
    def _majority_vote(self, tree_preds):
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        predictions = []
        for sample_preds in tree_preds:
            # Tìm nhãn được vote nhiều nhất
            most_common = np.bincount(sample_preds).argmax()
            predictions.append(most_common)
            
        return np.array(predictions)