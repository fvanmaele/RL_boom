import pickle

file = open("75-crate-no-opponents.pickle", "rb")
data = pickle.load(file)

print(data[0]['state'])
