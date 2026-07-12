# Screenshots Index

Evidence for MLOps Assignment 01 — Potturi Sai Sarath Chandra Murthy (2024AC05277)

## Included

| File | Shows |
|---|---|
| task1_missing_values_check.png | Missing-value counts: 4 in ca, 2 in thal |
| task1_cleaned_297_rows.png | Download/clean script output — 297 rows saved |
| eda_class_balance.png | 160 healthy vs 137 disease (46.1%) |
| eda_histograms.png | Numeric feature distributions |
| eda_correlation_heatmap.png | Feature correlations; thalach −0.42 with target |
| task2_training_cv_metrics.png | 5-fold CV metrics, best params C=1, ROC-AUC 0.917 |
| task3_training_with_mlflow.png | Training run with MLflow logging active |
| model_roc_curve_auc094.png | Final model hold-out ROC (AUC 0.94) — CI artifact |
| model_confusion_matrix_holdout.png | Hold-out confusion matrix (29/3/8/20) — CI artifact |
| task4_predict_output.png | Saved pipeline predicting from a dict |
| task4_clean_venv_install.png | Fresh venv install from pinned requirements.txt |
| task5_tests_8_passed.png | 8 pytest unit tests passing |
| task5_ci_red_run.png | Run #2: test job failed, train skipped (pipeline fails on errors) |
| task5_ci_green.png | Run #3: lint→test→train all green |
| task5_ci_artifacts.png | Workflow artifacts: trained-model + evaluation-plots |
| task6_docker_build.png | Image build success, tagged heart-api:latest |
| task6_docker_image_679mb.png | Docker Desktop: heart-api image, 679 MB |
| task6_docker_run.png | Container running, uvicorn startup complete |
| task6_api_docs.png | Swagger /docs served from the container |
| task6_docker_predict_200.png | POST /predict → 200 {"prediction":0,"confidence":0.862} |
| task7_pods_services.png | kubectl: 2 pods + LoadBalancer service (EXTERNAL-IP localhost) |
| repo_github_pushed.png | GitHub repository with full project structure |

## Still to capture (systems must be running)

- mlflow_runs_table.png — MLflow UI experiment with the 3 runs
- mlflow_artifacts_view.png — final run's Artifacts tab
- task7_k8s_predict.png — /predict served at http://localhost/docs (port 80)
- task7_pods_ready.png — kubectl get pods showing READY 1/1 (earlier capture was 0/1 at 18s)
- task8_metrics.png — /metrics with non-zero counters (endpoint must be built first)
- task8_request_logs.png — kubectl logs showing prediction log lines
- architecture_diagram.png — your draw.io export
