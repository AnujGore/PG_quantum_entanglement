from shadow import create_dataset, three_D_model

shadows, entropy = create_dataset(20, 2, 1)

plt = three_D_model(shadows, entropy)
plt.show()






