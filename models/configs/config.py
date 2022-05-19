# Dataset utils and sample selection
dataset_name = 'full_dataset_010322_nan_cleaned.csv'
select_one_level = True
select_level = 1

batch_size = 64
max_length = 300
training_epochs = 50
classes = 10
train_to_convergence = True
delta_threshold = 0.005
patience = 4
use_custom_learning_schedule = True
learning_rate = 0.05
delta_metric = ['acc']
