from mlflow.tracking.client import MlflowClient
import mlflow
import numpy as np


start_experiment=3
end_experiment=9
metric='val_accuracy'
num_epochs=50

#range(start_experiment,end_experiment+1)
experiment_ids=[1,2,3]


means=[]
stds=[]
for experiment_id in experiment_ids:
	experiment_id = str(experiment_id)
	experiment_name = mlflow.get_experiment(experiment_id).name
	print(experiment_name)
	results = mlflow.search_runs(experiment_ids=[experiment_id])

	max_accuracies=[]
	for run_id in (results['run_id']):
		val_accuracies=MlflowClient().get_metric_history(run_id, metric)
		max_accuracy=0.
		for epoch in range(num_epochs):
			accuracy=val_accuracies[epoch].value
			if val_accuracies[epoch].value > max_accuracy:
				max_accuracy=val_accuracies[epoch].value
		max_accuracies.append(max_accuracy)
	mean=np.mean(max_accuracies)
	std=np.std(max_accuracies)
	print('Mean:{} Std:{}'.format(mean, std))
	means.append(mean)
	stds.append(std)
	
#just use format

print("& 0 & ${:.3f}\pm{:.3f}$ & ${:.3f}\pm{:.3f}$ & ${:.3f}\pm{:.3f}${}".format(means[0], stds[0], means[1], stds[1], means[2], stds[2],r"\\"))
#60 mean, 60 std, 100 mean, 100 std, 150 mean, 150 std, 60 mean, 60 std, 100 mean, 100 std, 150 mean, 150 std
