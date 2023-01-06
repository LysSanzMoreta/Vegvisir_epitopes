"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import matplotlib.pyplot as plt
import seaborn as sns
def plot_heatmap(array, title,file_name):
    plt.figure(figsize=(20, 20))
    sns.heatmap(array, cmap='RdYlGn_r',yticklabels=False,xticklabels=False)
    plt.title(title)
    plt.savefig(file_name)
    plt.clf()
