import pickle

from matplotlib import pyplot as plt

with open("loss_and_r2.pickle", "rb") as f:
    losses, r2s = pickle.load(f)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.title("Loss")
plt.plot(losses)
plt.subplot(2, 1, 2)
plt.title("R2 Score")
plt.plot(r2s)
plt.savefig("fig.png")