{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cd73d849-c217-4f77-b741-194bc009553a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m44/44\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step\n",
      "Sample Predictions: [4.781019  4.7038417 4.340528  4.0862184 4.4728894]\n"
     ]
    }
   ],
   "source": [
    "# predict_magnitude.py\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "model = load_model('earthquake_magnitude_model.h5')\n",
    "X_test = np.load('x_test.npy')\n",
    "predictions = model.predict(X_test)\n",
    "print(\"Sample Predictions:\", predictions[:5].flatten())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f9fd898-8763-49dc-bf89-e01abd3b4241",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
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
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
