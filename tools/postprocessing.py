

def postprocess_predictions(predictions):
    predictions = predictions.copy()
    for p, pred in predictions.items():
        pred[pred < 0] = 0
        predictions[p] = pred
    l = predictions[0.1]
    m = predictions[0.5]
    u = predictions[0.9]
    idpredictions = l > m
    l[idpredictions] = m[idpredictions] - 0.001
    idpredictions = u < m
    u[idpredictions] = m[idpredictions] + 0.001
    predictions[0.1] = l
    predictions[0.5] = m
    predictions[0.9] = u
    return predictions