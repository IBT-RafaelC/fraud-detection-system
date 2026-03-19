{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "678cf084-fab5-497d-a37e-9dccfe2335ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    " \n",
    " \n",
    "def load_model(model_path=None):\n",
    "    \"\"\"Load model package from disk.\"\"\"\n",
    "    if model_path is None:\n",
    "        base_dir = Path(__file__).parent\n",
    "        model_path = base_dir / \"fraud_model.pkl\"\n",
    " \n",
    "    model_package = joblib.load(model_path)\n",
    "    return model_package[\"model\"], model_package[\"features\"]\n",
    " \n",
    " \n",
    "def make_prediction(data: pd.DataFrame, threshold: float = 0.5, model_path=None):\n",
    "    \"\"\"\n",
    "    Run fraud prediction on a DataFrame.\n",
    " \n",
    "    Args:\n",
    "        data: Input DataFrame with transaction features.\n",
    "        threshold: Probability cutoff for fraud classification.\n",
    "        model_path: Optional custom path to .pkl file.\n",
    " \n",
    "    Returns:\n",
    "        predictions (np.array), probabilities (np.array)\n",
    "    \"\"\"\n",
    "    model, features = load_model(model_path)\n",
    " \n",
    "    model_data = data.copy()\n",
    " \n",
    "    # Drop target/result columns if present\n",
    "    for col in ['Class', 'Fraud Prediction', 'Fraud Probability']:\n",
    "        if col in model_data.columns:\n",
    "            model_data = model_data.drop(col, axis=1)\n",
    " \n",
    "    # Validate features\n",
    "    missing = [f for f in features if f not in model_data.columns]\n",
    "    if missing:\n",
    "        raise ValueError(f\"Missing features: {missing}\")\n",
    " \n",
    "    model_data = model_data[features]\n",
    " \n",
    "    probabilities = model.predict_proba(model_data)[:, 1]\n",
    "    predictions = (probabilities >= threshold).astype(int)\n",
    " \n",
    "    return predictions, probabilities\n",
    " "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
