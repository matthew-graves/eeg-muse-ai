import brain_lib as bl
data = bl.get_dataset('data.h5')
data_only, labels = bl.separate_data(data)
model = bl.create_model()
bl.train_model(model, data_only, labels, 1000000)
print("All Done")
