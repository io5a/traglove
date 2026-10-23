# Machine learning

The checked-in inference artifacts are `models/svm_model_test_Cdata.pkl` and `models/scaler_Cdata.pkl`. The API applies the window feature transform in `software/api/features.py` before scaling and classifying. `models/knn_pipeline_best.pkl` supports the serial prediction client.

The KNN training script and instructions are in `training/`; its source recording is `data/data_new_2.csv`. The API's SVM model is supplied as a pre-trained artifact, while the serial client pipeline can be retrained. The serialized scikit-learn models were created with scikit-learn 1.8.0.
