{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "678cf084-fab5-497d-a37e-9dccfe2335ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\uvire\\anaconda3\\Lib\\site-packages\\pandas\\core\\arrays\\masked.py:60: UserWarning: Pandas requires version '1.3.6' or newer of 'bottleneck' (version '1.3.5' currently installed).\n",
      "  from pandas.core import (\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "def load_model():\n",
    "    model_package = joblib.load(\"C:/Users/uvire/Desktop/Proyectos IA y DS/fraud-detection-system/src/fraud_model.pkl\")\n",
    "    return model_package[\"model\"], model_package[\"features\"]\n",
    "\n",
    "def make_prediction(data):\n",
    "    model, features = load_model()\n",
    "\n",
    "    model_data = data.copy()\n",
    "\n",
    "    for col in ['Class', 'Fraud Prediction', 'Fraud Probability']:\n",
    "        if col in model_data.columns:\n",
    "            model_data = model_data.drop(col, axis=1)\n",
    "\n",
    "    model_data = model_data[features]\n",
    "\n",
    "    predictions = model.predict(model_data)\n",
    "    probabilities = model.predict_proba(model_data)[:,1]\n",
    "\n",
    "    return predictions, probabilities"
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
