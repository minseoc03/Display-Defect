from preprocess import preprocess
from optimize import optimize   
from train import train
from inference import inference

PATH = ''
SEED = 42
SEEDS_ENSEMBLE = [4872, 1231, 5231, 2313, 9821]

X, y, test = preprocess(PATH)
_, params = optimize(X, y, seed = SEED)
model, f1_score = train(X, y, params, seed = SEED)
test = inference(X, y, test, params, seeds = SEEDS_ENSEMBLE)

print(test['target'])
