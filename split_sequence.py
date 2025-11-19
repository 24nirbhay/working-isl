from src.data_collection import split_large_sequence

# Split the large hello sequence
split_large_sequence('good', 'data/dataset/good/sequence_0.csv', target_length=30, overlap=10)
