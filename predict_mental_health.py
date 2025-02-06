import pandas as pd
import numpy as np
import joblib
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

class MentalHealthInference:
    def __init__(self):
        # Paths to Google Drive datasets
        self.tech_survey_path = "/content/drive/MyDrive/survey.csv"
        self.depression_anxiety_path = "/content/drive/MyDrive/scores.csv"
        self.who_database_path = "/content/drive/MyDrive/mental.csv"
        
        # Load datasets
        self.tech_survey = pd.read_csv(self.tech_survey_path)
        self.depression_anxiety = pd.read_csv(self.depression_anxiety_path)
        self.who_data = pd.read_csv(self.who_database_path)
        
        # Load pre-trained model
        self.model = joblib.load("/content/drive/MyDrive/mental_health_model.pkl")
        
        # Prepare symptoms list
        self.symptoms_list = sorted(list(set(
            self.tech_survey['Symptom'].tolist() + 
            self.depression_anxiety['Symptom'].tolist() + 
            self.who_data['Symptom'].tolist()
        )))
    
    def predict_condition(self, user_input):
        """
        Predict mental health condition based on user input
        
        Args:
            user_input (str): User-reported symptoms
        
        Returns:
            tuple: (predicted condition, explanation, suggestion)
        """
        # Normalize input
        normalized_input = ' '.join(user_input.lower().split())
        
        # Create feature vector
        input_features = [1 if symptom.lower() in normalized_input else 0 for symptom in self.symptoms_list]
        
        # Check if any symptoms are detected
        if sum(input_features) == 0:
            return "No symptoms found", "Please describe symptoms more clearly", ""
        
        # Predict condition
        prediction = self.model.predict([input_features])[0]
        
        # Coping mechanisms
        coping_mechanisms = {
            "Anxiety": "Practice mindfulness and deep breathing techniques.",
            "Depression": "Consider professional counseling and journaling.",
            "Stress": "Implement stress management and seek social support."
        }
        
        return (
            prediction, 
            f"Prediction based on symptoms matching {sum(input_features)} of {len(self.symptoms_list)} known symptoms", 
            coping_mechanisms.get(prediction, "General support recommended")
        )
    
    def analyze_dataset_insights(self):
        """
        Provide insights from the loaded datasets
        
        Returns:
            dict: Dataset analysis summary
        """
        return {
            "Unique Symptoms": len(self.symptoms_list),
            "Tech Survey Symptoms": self.tech_survey['Symptom'].unique().tolist(),
            "Depression Anxiety Conditions": self.depression_anxiety['Condition'].unique().tolist(),
            "Total Data Points": len(self.tech_survey) + len(self.depression_anxiety) + len(self.who_data)
        }

# Example usage
if __name__ == "__main__":
    inference = MentalHealthInference()
    
    # Dataset insights
    print("Dataset Insights:")
    print(inference.analyze_dataset_insights())
    
    # Test predictions
    test_inputs = [
        "feeling tired and restless",
        "lack of sleep and anxiety",
        "depressed mood"
    ]
    
    for input_text in test_inputs:
        condition, explanation, suggestion = inference.predict_condition(input_text)
        print(f"\nInput: {input_text}")
        print(f"Predicted Condition: {condition}")
        print(f"Explanation: {explanation}")
        print(f"Suggestion: {suggestion}")
